import os
import pytz
import torch
import wandb
import numpy as np
import argparse
import json

import mdtraj as md
from datetime import datetime

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, BoltzmannGeneratorCV, MeanFreeNormalDistribution
from tbg.modelwithcv import EGNN_AD2_CV
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm

ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]


def coordinate2distance(
    positions
):
    '''
        Compute pairwise distances between heavy atoms of alanine
        Args:
            positions (torch.Tensor): Positions of atoms (n_samples, *)
    '''
    
    distance_list = []
    
    for position in positions:
        position = position.reshape(-1, 3)
        heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
        num_heavy_atoms = len(heavy_atom_position)
        distance = []
        for i in range(num_heavy_atoms):
            for j in range(i+1, num_heavy_atoms):
                distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
        distance = torch.stack(distance)
        distance_list.append(distance)
    
    return torch.stack(distance_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--data_xyz_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg.pt", help='Path to xyz data file')
    parser.add_argument('--data_distance_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt", help='Path to distance data file')
    parser.add_argument('--data_type', type=str, default="10n", help='data type')
    parser.add_argument('--sample_epoch', type=int, default="500", help='epoch interval for sampling')
    parser.add_argument('--n_epochs', type=int, default="1000", help='Number of epochs to train')
    parser.add_argument('--sigma', type=float, default="0", help='Sigma value for CNF')
    parser.add_argument('--hidden_dim', type=int, default="256", help='hidden dimension of EGNN')
    parser.add_argument('--lambda_state_diff', type=float, default="0.5", help='Lambda for state difference loss')
    parser.add_argument('--tags', nargs='*', help='Tags for Wandb')
    
    return parser.parse_args()

args = parse_args()

n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
atom_types = np.arange(22)
# atom_types[[1, 2, 3]] = 2
atom_types[[0, 2, 3]] = 0
atom_types[1] = 2
atom_types[[11, 12, 13]] = 12
atom_types[[19, 20, 21]] = 20
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


wandb.init(
    project="tbg",
    entity="eddy26",
    config=vars(args),
    tags=["condition", "ECNF++"] + args.tags,
)


prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)
brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_AD2_CV(
    n_particles=n_particles,
    device="cuda",
    n_dimension=dim // n_particles,
    h_initial=h_initial,
    hidden_nf=args.hidden_dim,
    act_fn=torch.nn.SiLU(),
    n_layers=5,
    recurrent=True,
    tanh=True,
    attention=True,
    condition_time=True,
    mode="egnn_dynamics",
    agg="sum",
)
bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator
)
flow = DiffEqFlow(dynamics=bb_dynamics)
flow._dynamics._dynamics._dynamics_function.cv.condition = True
bg = BoltzmannGeneratorCV(prior, flow, prior).cuda()


n_batch = 256
data_xyz = torch.load(args.data_xyz_path).cuda()
data_distance = torch.load(args.data_distance_path).cuda()
batch_iter = IndexBatchIterator(len(data_xyz), n_batch)
optim = torch.optim.AdamW(flow.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler = CosineAnnealingWarmRestarts(optim, T_0=args.n_epochs, eta_min=1e-6)
kst = pytz.timezone('Asia/Seoul')
now = datetime.now(kst)
folder_name = now.strftime('%m%d-%H%M%S')
PATH_last = f"models/tbgcv/{folder_name}/"
if not os.path.exists(PATH_last):
    os.makedirs(PATH_last)
else:
    raise ValueError(f"Folder {PATH_last} already exists")
# Save args to JSON file
with open(os.path.join(PATH_last, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

c5_state = torch.tensor(md.load("../../simulation/data/alanine/c5.pdb").xyz, device="cuda")
c7ax_state = torch.tensor(md.load("../../simulation/data/alanine/c7ax.pdb").xyz, device="cuda")
c5_heavy_atom_distance = coordinate2distance(c5_state)
c7ax_heavy_atom_distance = coordinate2distance(c7ax_state)


sigma = args.sigma
epoch_loss = torch.tensor(0.0).cuda()
pbar = tqdm(range(args.n_epochs), desc = f"Loss: {epoch_loss:.4f}",)
for epoch in pbar:
    loss_list = []
    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        # Load data
        x1 = data_xyz[idx][:, 1]
        x1_distance = data_distance[idx][:, 0]
        batchsize = x1.shape[0]
        t = torch.rand(batchsize, 1).cuda()
        x0 = prior_cpu.sample(batchsize).cuda()

        # calculate regression loss
        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        noise = prior.sample(batchsize)
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        
        # Flow
        # vt = flow._dynamics._dynamics._dynamics_function(t, x)
        vt = flow._dynamics._dynamics._dynamics_function(t, x, x1_distance)
        loss = torch.mean((vt - ut) ** 2)
        
        # State difference
        c5_mlcv = flow._dynamics._dynamics._dynamics_function.cv(c5_heavy_atom_distance)
        c7ax_mlcv = flow._dynamics._dynamics._dynamics_function.cv(c7ax_heavy_atom_distance)
        state_diff = torch.norm(c5_mlcv - c7ax_mlcv)
        loss -= args.lambda_state_diff * state_diff
        
        loss_list.append(loss.item())
        loss.backward()
        optim.step()
        scheduler.step(epoch + it / len(batch_iter))
    epoch_loss = np.mean(loss_list)
    pbar.set_description(f"Loss: {epoch_loss:.4f}")
    
    wandb.log({
        "loss": epoch_loss,
        "loss/state_diff": state_diff,
        "lr": scheduler.get_last_lr()[0]
    }, step=epoch)
    
    if epoch % 500 == 0 and epoch != 0:
        print(f"Epoch {epoch}")
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last + f"/tbg_{epoch}.pt",
        )
        torch.save(
            flow._dynamics._dynamics._dynamics_function.cv.state_dict(),
            PATH_last + f"/mlcv_{epoch}.pt",   
        )
        
print(f">> Final epoch {epoch}")
torch.save(
    {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
    },
    PATH_last + f"/tbg_{epoch}.pt",
)
torch.save(
    flow._dynamics._dynamics._dynamics_function.cv.state_dict(),
    PATH_last + f"/mlcv_{epoch}.pt",   
)


wandb.finish()