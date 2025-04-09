import os
import pytz
import torch
import wandb
import numpy as np

from tqdm.auto import tqdm
from datetime import datetime

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, MeanFreeNormalDistribution
from tbg.models2 import EGNN_dynamics_AD2_cat
from bgflow import BlackBoxDynamics, BruteForceEstimator


n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions


# atom types for backbone
atom_types = np.arange(22)
atom_types[[1, 2, 3]] = 2
atom_types[[19, 20, 21]] = 20
atom_types[[11, 12, 13]] = 12
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))

wandb.init(
    project="tbg",
    entity="eddy26",
    tags=["repro"],
)
kst = pytz.timezone('Asia/Seoul')
now = datetime.now(kst)
folder_name = now.strftime('%m%d-%H%M%S')
PATH_last = f"models/repro/{folder_name}"
if not os.path.exists(PATH_last):
    os.makedirs(PATH_last)


# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)

brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_dynamics_AD2_cat(
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
# data_path = "data/AD2/AD2_weighted.npy"
# data_smaller = torch.from_numpy(np.load(data_path)).float()
data_xyz = torch.load("../../simulation/dataset/alanine/300.0/tbg-10n/current-xyz.pt").cuda()
batch_iter = IndexBatchIterator(len(data_xyz), n_batch)
optim = torch.optim.Adam(flow.parameters(), lr=5e-4)
n_epochs = 1000
sigma = 0.01
epoch_loss = torch.tensor(0.0).cuda()

pbar = tqdm(range(n_epochs), desc = f"Loss: {epoch_loss:.4f}",)
for epoch in range(n_epochs):
    if epoch == 500:
        for g in optim.param_groups:
            g["lr"] = 5e-5
    loss_list = []
    
    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

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
        vt = flow._dynamics._dynamics._dynamics_function(t, x)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optim.step()
        loss_list.append(loss.item())
        
    epoch_loss = np.mean(loss_list)
    pbar.set_description(f"Loss: {epoch_loss:.4f}")
    pbar.update(1)
    wandb.log({
        "loss": epoch_loss,
    }, step=epoch)    
    
    if epoch > 0 and epoch % 400 == 0:
        print(epoch)
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last + f"/{epoch}.pt",
        )
        

torch.save(
    {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
    },
    PATH_last + "/tbg.pt",
)