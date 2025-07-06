import torch
import torch.nn as nn

from tbg.gcl import E_GCL_vel, E_GCL, GCL, ZeroLinear
from tbg.utils import remove_mean, remove_mean_with_mask


CV_DIMENSION = 2

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
    
    
class EGNN_CV_CONTROL(nn.Module):
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
            
            self.egnn = EGNN_CONTROL(in_node_nf=h_size, in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
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
            _, x_final = self.egnn(h, x, edges, cv_condition=cv_condition, edge_attr=edge_attr)
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
    
    def load_pretrained(self, path_to_pretrained_model):       
        state_dict = torch.load(path_to_pretrained_model)["model_state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if "egnn." in key:
                stripped_key = key.split("egnn.", 1)[-1]
                new_state_dict[stripped_key] = value
                new_state_dict["control_" + stripped_key] = value
            else:
                new_state_dict[key] = value  # keep untouched keys
        
        self.egnn.load_state_dict(new_state_dict)
        
        for name, param in self.egnn.named_parameters():
            if name.startswith("control_"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for i in range(0, self.egnn.n_layers):
            self.egnn.add_module("zero_pre_h_%d" % i, ZeroLinear(self.egnn.hidden_nf, self.egnn.hidden_nf))
            self.egnn.add_module("zero_post_h_%d" % i, ZeroLinear(self.egnn.hidden_nf, self.egnn.hidden_nf))
            self.egnn.add_module("zero_pre_x_%d" % i, ZeroLinear(CV_DIMENSION, 3))
            self.egnn.add_module("zero_post_x_%d" % i, ZeroLinear(3, 3))
        self.egnn.to(self.device)
        
        return
    

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
    
class EGNN_CONTROL(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum'):
        super(EGNN_CONTROL, self).__init__()
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

        # Add trainable layer wise controlnet, and freze original
        self.add_module("control_embedding", nn.Linear(in_node_nf, self.hidden_nf))
        self.add_module("control_embedding_out", nn.Linear(self.hidden_nf, out_node_nf))
        for i in range(0, n_layers):
            self.add_module("control_gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer, agg=agg))
        
        self.to(self.device)

    def forward(self, h, x, edges, cv_condition=None, edge_attr=None, node_mask=None, edge_mask=None):
        h_in = h
        x_in = x
        h = self.embedding(h)
        
        # TODO: input for controller
        cv_condition_repeated = torch.repeat_interleave(cv_condition, 22, dim=0).reshape(-1, CV_DIMENSION)
        control_h = h
        control_x = x
        
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            control_h = self._modules["zero_pre_h_%d" % i](control_h)
            control_x = self._modules["zero_pre_x_%d" % i](cv_condition_repeated) + control_x
            control_h, control_x, _ = self._modules["control_gcl_%d" % i](control_h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            control_h = self._modules["zero_post_h_%d" % i](control_h)
            control_x = self._modules["zero_post_x_%d" % i](control_x)
            h = h + control_h
            x = x + control_x
        
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

