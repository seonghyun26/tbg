import os
import pytz
import torch
import wandb
import numpy as np
import argparse

from datetime import datetime

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, MeanFreeNormalDistribution
from tbg.models2 import EGNN_dynamics_AD2_cat
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--data_xyz_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg2.pt", help='Path to xyz data file')
    parser.add_argument('--data_distance_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt", help='Path to distance data file')
    parser.add_argument('--condition_scaling', type=float, default="6", help='Scaling for conditioning')
    parser.add_argument('--n_epochs', type=int, default="1000", help='Number of epochs to train')
    parser.add_argument('--n_batch', type=int, default="256", help='Data batch size')
    parser.add_argument('--sigma', type=float, default="0.01", help='Sigma value for CNF')
    parser.add_argument('--hidden_dim', type=int, default="64", help='hidden dimension of EGNN')
    parser.add_argument('--ckpt_name', type=str, default="ckpt", help='Checkpoint name')
    parser.add_argument('--repro', type=bool, default=False, help='Reproduction settings')
    parser.add_argument('--cfg', type=bool, default=False, help='CFG boolean flag')
    parser.add_argument('--cfg_p', type=float, default=0.2, help='Threshold for CFG')
    parser.add_argument('--tags', nargs='*', help='Tags for Wandb')
    
    return parser.parse_args()

args = parse_args()


wandb.init(
    project="tbg",
    entity="eddy26",
    config=vars(args),
    tags=["reproduce", "custom-data"]
)
kst = pytz.timezone('Asia/Seoul')
now = datetime.now(kst)
folder_name = now.strftime('%m%d-%H%M%S')
PATH_last = f"models/repro/{folder_name}/"
if not os.path.exists(PATH_last):
    os.makedirs(PATH_last)
else:
    raise ValueError(f"Folder {PATH_last} already exists")


# atom types for backbone
n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
atom_types = np.arange(22)
# atom_types[[0, 2, 3]] = 1
atom_types[[1, 2, 3]] = 2
atom_types[[19, 20, 21]] = 20
atom_types[[11, 12, 13]] = 12
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)
brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_dynamics_AD2_cat(
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
    dynamics_function=net_dynamics,
    divergence_estimator=brute_force_estimator
)
flow = DiffEqFlow(dynamics=bb_dynamics)


n_batch = 256
# data_path = "data/AD2/AD2_weighted.npy"
# data_xyz = torch.from_numpy(np.load(data_path)).float()
data_xyz = torch.load(args.data_xyz_path)[:, 0]
batch_iter = IndexBatchIterator(len(data_xyz), n_batch)
optim = torch.optim.Adam(flow.parameters(), lr=5e-4)
n_epochs = 1000


sigma = 0.01
loss = torch.tensor(0.0).cuda()
pbar = tqdm(range(n_epochs), desc = f"Loss: {loss:.4f}",)
for epoch in pbar:
    if epoch == 500:
        for g in optim.param_groups:
            g["lr"] = 5e-5

    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        # x1 = data_smaller[idx].cuda()
        x1 = data_xyz[idx].cuda()
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
        vt = flow._dynamics._dynamics._dynamics_function(t, x)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optim.step()
    pbar.set_description(f"Loss: {loss:.4f}")
    
    wandb.log({
        "loss": loss.item(),
        "lr": optim.param_groups[0]["lr"],
    }, step=epoch)
    
    if epoch and epoch % 400 == 0:
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last + f"/tbg_{epoch}.pt",
        )

print(f">> Final epoch {epoch}")
torch.save(
    {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
    },
    PATH_last + f"/tbg-{args.ckpt_name}.pt",
)
print(f"Model saved to {PATH_last}")
wandb.save(PATH_last + f"/tbg-{args.ckpt_name}.pt")

wandb.finish()