import torch
import torch.nn as nn

from tbg.gcl import E_GCL_vel, E_GCL, GCL
from tbg.utils import remove_mean, remove_mean_with_mask



class EGNN_dynamics(nn.Module):
    def __init__(self, n_particles, n_dimension, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(in_node_nf=1, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(in_node_nf=1 + n_dimension, in_edge_nf=0, hidden_nf=hidden_nf, out_node_nf=n_dimension, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention)

        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        # Count function calls
        self.counter = 0

    def forward(self, t, xs):

        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        #Changed by Leon
        x = xs.reshape(n_batch*self._n_particles, self._n_dimension).clone()
        h = torch.ones(n_batch, self._n_particles).to(self.device)
    
        if self.condition_time:
            h = h*t
        h = h.reshape(n_batch*self._n_particles, 1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x

        elif self.mode == 'gnn_dynamics':
            h = torch.cat([h, x], dim=1)
            vel = self.gnn(h, edges)

        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel)
        #Changed by Leon
        self.counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]
    
class EGNN_AD2_CV(nn.Module):
    def __init__(self, n_particles, n_dimension,h_initial, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, cv_dimension=0, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.cv_dimension = cv_dimension
        
        # Initial one hot encoding of the different element types
        self.h_initial = h_initial

        if mode == 'egnn_dynamics':
            h_size = h_initial.size(1)
            if condition_time:
                h_size += 1
            if cv_dimension:
                h_size += self.cv_dimension
            
            self.egnn = EGNN(in_node_nf=h_size, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        else:
            raise NotImplemented()


        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        
        # Count function calls
        self.counter = 0


    def forward(self, t, xs, cv_condition=None):
        # Batch
        xs = xs[:, :self._n_particles * 3]
        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        x = xs.reshape(n_batch * self._n_particles, self._n_dimension).clone()
        
        
        # node feature
        h = self.h_initial.to(self.device).reshape(1,-1)
        h = h.repeat(n_batch, 1)
        h = h.reshape(n_batch * self._n_particles, -1)
        
        if self.cv_dimension and cv_condition is not None:
            cv_condition_molecule = cv_condition.repeat_interleave(self._n_particles, dim=0)
            h = torch.cat([h, cv_condition_molecule], dim=1)
        
        # Time
        t = torch.as_tensor(t, device=xs.device)
        if t.shape != (n_batch,1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch*self._n_particles, 1)
        
        # NN
        if self.condition_time:
            h = torch.cat([h, t], dim=-1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x

        else:
            raise NotImplemented()
        
        
        
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel)
        
        #print(t, xs)
        self.counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]
    


class Finetune_MLP(nn.Module):
    def __init__(self, h_size, cv_dimension):
        super(Finetune_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h_size + cv_dimension, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, h_size),
        )

    def forward(self, x):
        return self.mlp(x)

class EGNN_AD2_CV_FINETUNE(nn.Module):
    def __init__(
            self, n_particles, n_dimension,h_initial, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
            condition_time=True, cv_dimension=0, tanh=False, mode='egnn_dynamics', agg='sum',
        ):
        super().__init__()
        self.mode = mode
        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.cv_dimension = cv_dimension
        
        # Initial one hot encoding of the different element types
        self.h_initial = h_initial

        if mode == 'egnn_dynamics':
            h_size = h_initial.size(1)
            if condition_time:
                h_size += 1
            
            self.finetune_mlp = Finetune_MLP(h_size, cv_dimension)
            self.egnn = EGNN(in_node_nf=h_size, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        else:
            raise NotImplemented()


        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        
        # Count function calls
        self.counter = 0


    def forward(self, t, xs, cv_condition):
        # Batch
        xs = xs[:, :self._n_particles * 3]
        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        x = xs.reshape(n_batch * self._n_particles, self._n_dimension).clone()
        
        
        # node feature
        h = self.h_initial.to(self.device).reshape(1,-1)
        h = h.repeat(n_batch, 1)
        h = h.reshape(n_batch * self._n_particles, -1)
        
        if self.finetune_mlp is not None:
            cv_condition_molecule = cv_condition.repeat_interleave(self._n_particles, dim=0)
            h = self.finetune_mlp(torch.cat([h, cv_condition_molecule], dim=1))
        
        # Time
        t = torch.as_tensor(t, device=xs.device)
        if t.shape != (n_batch,1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch*self._n_particles, 1)
        
        # NN
        if self.condition_time:
            h = torch.cat([h, t], dim=-1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x

        else:
            raise NotImplemented()
        
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel)
        
        self.counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]
    
    
class EGNN_AD2_CFG(nn.Module):
    def __init__(self, n_particles, n_dimension,h_initial, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False,
                 condition_time=True, cv_dimension=0, condition_cv=True, tanh=False, mode='egnn_dynamics', agg='sum'):
        super().__init__()
        self.mode = mode
        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.cv_dimension = cv_dimension 
        
        # Initial one hot encoding of the different element types
        self.h_initial = h_initial

        if mode == 'egnn_dynamics':
            h_size = h_initial.size(1)
            if condition_time:
                h_size += 1
            if condition_cv:
                h_size += self.cv_dimension
            
            self.egnn = EGNN(in_node_nf=h_size, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        else:
            raise NotImplemented()


        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        
        # Count function calls
        self.counter = 0


    def forward(self, t, xs, cv_condition=None):
        # Batch
        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        x = xs.reshape(n_batch * self._n_particles, self._n_dimension).clone()
        
        
        # node feature
        h = self.h_initial.to(self.device).reshape(1,-1)
        h = h.repeat(n_batch, 1)
        h = h.reshape(n_batch * self._n_particles, -1)
        
        # CV condition with CFG
        if self.cv_dimension and cv_condition is not None:
            cv_condition_molecule = cv_condition.repeat_interleave(self._n_particles, dim=0)
            h = torch.cat([h, cv_condition_molecule], dim=1)
        
        # Time
        t = torch.as_tensor(t, device=xs.device)
        if t.shape != (n_batch,1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch*self._n_particles, 1)
        
        # NN
        if self.condition_time:
            h = torch.cat([h, t], dim=-1)
        if self.mode == 'egnn_dynamics':
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
            vel = x_final - x
        else:
            raise NotImplemented()
            
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel)
        
        # TODO: samplign with guidance
        
        #print(t, xs)
        self.counter += 1
        return vel.view(n_batch,  self._n_particles* self._n_dimension)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer, agg=agg))

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention))

        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h

