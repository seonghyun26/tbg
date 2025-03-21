import os
import torch
import wandb
import numpy as np
import argparse

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, MeanFreeNormalDistribution
from tbg.modelwithcv import EGNN_AD2_CFG
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--data_xyz_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg.pt", help='Path to xyz data file')
    parser.add_argument('--data_distance_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt", help='Path to distance data file')
    parser.add_argument('--data_type', type=str, default= "1n", help='data type')
    return parser.parse_args()

n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions

args = parse_args()

wandb.init(
    project="tbg",
    entity="eddy26",
    config={
        "data": args.data_xyz_path,
    },
    tags=["both"]
)

# atom types for backbone
atom_types = np.arange(22)
atom_types[[1, 2, 3]] = 2
atom_types[[19, 20, 21]] = 20
atom_types[[11, 12, 13]] = 12
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)

brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_AD2_CFG(
    n_particles=n_particles,
    device="cuda",
    n_dimension=dim // n_particles,
    h_initial=h_initial,
    hidden_nf=64,
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


n_batch = 256
data_xyz_path = args.data_xyz_path
data_xyz = torch.load(data_xyz_path)
data_distance_path = args.data_distance_path
data_distance = torch.load(data_distance_path)
batch_iter = IndexBatchIterator(len(data_xyz), n_batch)
# Dataset size: data num * (current, time lag, distance, time lag distance) * coordinates (66)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)
n_epochs = 1000
PATH_last = f"models/tbgcv-both/{args.data_type}"
if not os.path.exists(PATH_last):
    os.makedirs(PATH_last)
PATH_last = f"models/tbgcv-both/{args.data_type}/"


sigma = 0.01
for epoch in tqdm(range(n_epochs)):
    if epoch == 500:
        for g in optim.param_groups:
            g["lr"] = 5e-5

    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        # data
        x1_current = data_xyz[idx][:, 0].cuda()
        x1_timelag = data_xyz[idx][:, 1].cuda()
        x1_distance = data_distance[idx][:, 0].cuda()
        batchsize = x1_current.shape[0]
        
        # condition and non-condition
        p_uncond = torch.rand(batchsize).cuda()
        uncond_mask = p_uncond < 0.2
        x1  = torch.where(uncond_mask[:, None], x1_current, x1_timelag)
        cv = flow._dynamics._dynamics._dynamics_function.cv(x1_distance)
        cv_condition = torch.where(uncond_mask[:, None], torch.zeros_like(cv), cv)
        
        # prior
        t = torch.rand(batchsize, 1).cuda()
        x0 = prior_cpu.sample(batchsize).cuda()

        # calculate regression loss
        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        noise = prior.sample(batchsize)
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        
        # Flow
        # flow._dynamics._dynamics._dynamics_function.cv.condition = False
        x = torch.cat([x, cv_condition], dim=1)
        vt = flow._dynamics._dynamics._dynamics_function(t, x)
        loss = torch.mean((vt - ut) ** 2)
        
        loss.backward()
        optim.step()
    
    wandb.log({
        "loss": loss.item(),
    }, step=epoch)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}")
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last + f"/_tbg_{epoch}.pt",
        )
        torch.save(
            flow._dynamics._dynamics._dynamics_function.cv.state_dict(),
            PATH_last + f"/_mlcv_{epoch}.pt",   
        )

torch.save(
    {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
    },
    PATH_last + f"/_tbg_{epoch}.pt",
)
torch.save(
    flow._dynamics._dynamics._dynamics_function.cv.state_dict(),
    PATH_last + f"/_mlcv_{epoch}.pt",   
)

wandb.finish()