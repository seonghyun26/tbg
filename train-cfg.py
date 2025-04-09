import os
import pytz
import torch
import wandb
import numpy as np
import argparse

from datetime import datetime

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, MeanFreeNormalDistribution
from tbg.modelwithcv import EGNN_AD2_CV
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm

# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.scheduler import CosineAnnealingWarmUpRestarts


def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--data_xyz_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg.pt", help='Path to xyz data file')
    parser.add_argument('--data_distance_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt", help='Path to distance data file')
    parser.add_argument('--data_type', type=str, default="10n", help='data type')
    parser.add_argument('--sample_epoch', type=int, default="500", help='epoch interval for sampling')
    parser.add_argument('--n_epochs', type=int, default="1000", help='Number of epochs to train')
    parser.add_argument('--sigma', type=float, default="0", help='Sigma value for CNF')
    parser.add_argument('--hidden_dim', type=int, default="256", help='hidden dimension of EGNN')
    parser.add_argument('--tags', nargs='*', help='Tags for Wandb')
    
    return parser.parse_args()
    
args = parse_args()


n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
atom_types = np.arange(22)
# atom_types[[0, 2, 3]] = 0
# atom_types[1] = 2
atom_types[[1, 2, 3]] = 2
atom_types[[11, 12, 13]] = 12
atom_types[[19, 20, 21]] = 20
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


wandb.init(
    project="tbg",
    entity="eddy26",
    config=vars(args),
    tags=["cfg", "ECNF++"] + args.tags,
)

# Set saving configs
kst = pytz.timezone('Asia/Seoul')
now = datetime.now(kst)
folder_name = now.strftime('%m%d-%H%M%S')
PATH_last = f"models/tbgcv/{folder_name}/"
if not os.path.exists(PATH_last):
    os.makedirs(PATH_last)
else:
    raise ValueError(f"Folder {PATH_last} already exists")

# Set wandb configs



prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)
brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_AD2_CV(
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
    dynamics_function=net_dynamics,
    divergence_estimator=brute_force_estimator
)
flow = DiffEqFlow(dynamics=bb_dynamics)

# Set dataset and optimizer configs
n_batch = 256
data_xyz = torch.load(args.data_xyz_path).cuda()
data_distance = torch.load(args.data_distance_path).cuda()
batch_iter = IndexBatchIterator(len(data_xyz), n_batch)
# Dataset size: data num * (current, time lag, distance, time lag distance) * coordinates (66)
# optim = torch.optim.Adam(flow.parameters(), lr=5e-4)
optim = torch.optim.AdamW(flow.parameters(), lr=1e-6, weight_decay=1e-2)
scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=args.n_epochs, eta_max=5e-4)

sigma = args.sigma
loss = torch.tensor(0.0).cuda()
pbar = tqdm(range(args.n_epochs), desc = f"Loss: {loss:.4f}",)
for epoch in pbar:
    loss_list = []
    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        # Load data
        x1_current = data_xyz[idx][:, 0]
        x1_timelag = data_xyz[idx][:, 1]
        x1_distance = data_distance[idx][:, 0]
        batchsize = x1_current.shape[0]
        
        # condition and non-condition
        p_uncond = torch.rand(batchsize).cuda()
        uncond_mask = p_uncond < 0.2
        x1 = torch.where(uncond_mask[:, None], x1_current, x1_timelag)
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
        vt = flow._dynamics._dynamics._dynamics_function(t, x, cv_condition)
        loss = torch.mean((vt - ut) ** 2)
        loss_list.append(loss.item())
        loss.backward()
        optim.step()
        scheduler.step(epoch + it / len(batch_iter))
    epoch_loss = np.mean(loss_list)
    pbar.set_description(f"Loss: {loss:.4f}")
    
    
    wandb.log({
        "loss": loss.item(),
        "lr": scheduler.get_last_lr()[0]
    }, step=epoch)
    
    if epoch and epoch % 500 == 0:
        print(f"Epoch {epoch}")
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last + f"/tbgcfg_{epoch}.pt",
        )
        torch.save(
            flow._dynamics._dynamics._dynamics_function.cv.state_dict(),
            PATH_last + f"/mlcvcfg_{epoch}.pt",   
        )

torch.save(
    {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
    },
    PATH_last + f"/tbgcfg_{epoch}.pt",
)
torch.save(
    flow._dynamics._dynamics._dynamics_function.cv.state_dict(),
    PATH_last + f"/mlcvcfg_{epoch}.pt",   
)

wandb.finish()