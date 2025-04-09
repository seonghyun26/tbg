import os
import pytz
import torch
import wandb
import numpy as np
import argparse

from datetime import datetime

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, BoltzmannGeneratorCV, MeanFreeNormalDistribution
from tbg.modelwithcv import EGNN_AD2_CV, TBGCV
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm

# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.scheduler import CosineAnnealingWarmUpRestarts


def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--current_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/current-xyz.pt", help='Path to current xyz data file')
    parser.add_argument('--timelag_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/timelag-xyz.pt", help='Path to timelag xyz data file')
    parser.add_argument('--current_distance', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/current-distance.pt", help='Path to current distance data file')
    parser.add_argument('--n_epochs', type=int, default="1000", help='Number of epochs to train')
    parser.add_argument('--n_batch', type=int, default="256", help='Data batch size')
    parser.add_argument('--sigma', type=float, default="0.00", help='Sigma value for CNF')
    parser.add_argument('--hidden_dim', type=int, default="64", help='hidden dimension of EGNN')
    parser.add_argument('--type', type=str, default="cv-condition", help='training type')
    parser.add_argument('--cfg_p', type=float, default=0.2, help='Threshold for CFG')
    parser.add_argument('--ckpt_name', type=str, help='Checkpoint name')
    parser.add_argument('--tags', nargs='*', help='Tags for Wandb')
    parser.add_argument('--cv_condition_scale', type=float, default="6", help='Scaling for cv condition')
    
    return parser.parse_args()

args = parse_args()

n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
atom_types = np.arange(22)
atom_types[[1, 2, 3]] = 2
atom_types[[11, 12, 13]] = 12
atom_types[[19, 20, 21]] = 20
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


wandb.init(
    project="tbg",
    entity="eddy26",
    config=vars(args),
    tags=["condition", "ECNF++"] + args.tags,
)
kst = pytz.timezone('Asia/Seoul')
now = datetime.now(kst)
folder_name = now.strftime('%m%d-%H%M%S')
if args.type in ["repro", "label"]:
    PATH_last = f"models/tbgcv-repro/{folder_name}/"
elif args.type == "cfg":
    PATH_last = f"models/tbgcv-cfg/{folder_name}/"
elif args.type == "cv-condition":
    PATH_last = f"models/tbgcv/{folder_name}/"
else:
    raise ValueError(f"Unknown training type {args.type}")
if not os.path.exists(PATH_last):
    os.makedirs(PATH_last)
else:
    raise ValueError(f"Folder {PATH_last} already exists")
print(f"Model will be saved to {PATH_last}")


if args.type == "cv-condition":
    encoder_layers = [45, 30, 30, 2]
    cv_dimension = encoder_layers[-1]
    tbgcv = TBGCV(encoder_layers=encoder_layers).cuda()
    tbgcv.train()
elif args.type == "label":
    cv_dimension = 22
else:
    cv_dimension = 0


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
    cv_dimension=cv_dimension,
    mode="egnn_dynamics",
    agg="sum",
)
bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator
)
flow = DiffEqFlow(dynamics=bb_dynamics)


# Load dataset, set up optimizer
data_current_xyz = torch.load(args.current_xyz).cuda()
data_timelag_xyz = torch.load(args.timelag_xyz).cuda()
data_current_distance = torch.load(args.current_distance).cuda()
data_label = torch.load("../../simulation/dataset/alanine/300.0/tbg-10n/current-label.pt").cuda()
batch_iter = IndexBatchIterator(len(data_current_xyz), args.n_batch)
optim = torch.optim.AdamW(flow.parameters(), lr=1e-6, weight_decay=1e-3)
scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=args.n_epochs, T_up=50, eta_max=5e-4, gamma=0.5)


epoch_loss = torch.tensor(0.0).cuda()
pbar = tqdm(range(args.n_epochs), desc = f"Loss: {epoch_loss:.4f}",)
for epoch in pbar:
    loss_list = []
    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        # Load data
        x1_current = data_current_xyz[idx]
        x1_timelag = data_timelag_xyz[idx]
        x1_label = data_label[idx]
        x1_distance = data_current_distance[idx]
        batchsize = x1_current.shape[0]
        
        # type
        if args.type == "repro":
            x1 = x1_current
            cv_condition = None
        elif args.type == "label":
            x1 = x1_current
            cv_condition = x1_label.unsqueeze(1).repeat(1, cv_dimension)
        elif args.type == "cfg":
            uncond_mask = torch.rand(batchsize).cuda() < args.cfg_p
            x1 = torch.where(uncond_mask[:, None], x1_current, x1_timelag)
            cv_condition = tbgcv(x1_distance)
            cv_condition = torch.where(uncond_mask[:, None], torch.zeros_like(cv_condition), cv_condition)
        elif args.type == "cv-condition":
            x1 = x1_timelag
            cv_condition = tbgcv(x1_distance)
        else:
            raise ValueError(f"Unknown training type {args.type}")

        # calculate regression loss
        t = torch.rand(batchsize, 1).cuda()
        # x0 = prior_cpu.sample(batchsize).cuda()
        x0 = prior.sample(batchsize)
        mu_t = x0 * (1 - t) + x1 * t
        noise = prior.sample(batchsize)
        x = mu_t + args.sigma * noise
        ut = x1 - x0
        
        # Flow
        # vt = flow._dynamics._dynamics._dynamics_function(t, x)
        vt = flow._dynamics._dynamics._dynamics_function(t, x, cv_condition)
        loss = torch.mean((vt - ut) ** 2)
        loss_list.append(loss.item())
        loss.backward()
        optim.step()
        scheduler.step(epoch + it / len(batch_iter))
    epoch_loss = np.mean(loss_list)
    pbar.set_description(f"Loss: {epoch_loss:.4f}")
    
    wandb.log({
        "loss": epoch_loss,
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
        if args.type not in ["repro", "label"]:    
            torch.save(
                tbgcv.state_dict(),
                PATH_last + f"/mlcv_{epoch}.pt",   
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
wandb.save(PATH_last + f"/tbg-{args.ckpt_name}.pt")

if args.type not in ["repro", "label"]:
    torch.save(
        tbgcv.state_dict(),
        PATH_last + f"/mlcv-{args.ckpt_name}.pt",   
    )
    wandb.save(PATH_last + f"/mlcv-{args.ckpt_name}.pt")
print(f"Model saved to {PATH_last}")

wandb.finish()