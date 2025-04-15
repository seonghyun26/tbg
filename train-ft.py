import os
import json
import torch
import wandb
import numpy as np
import argparse

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, MeanFreeNormalDistribution
from tbg.modelwithcv import EGNN_AD2_CV
from tbg.cv import TBGCV
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm

# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.scheduler import CosineAnnealingWarmUpRestarts


PHI_ANGLE = [4, 6, 8, 14]
PSI_ANGLE = [6, 8, 14, 16]
ALANINE_HEAVY_ATOM_IDX = [0, 4, 5, 6, 8, 10, 14, 15, 16, 18]



def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--date', type=str, default= "debug", help='Date for the experiment')
    parser.add_argument('--current_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/current-xyz.pt", help='Path to current xyz data file')
    parser.add_argument('--timelag_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/timelag-xyz.pt", help='Path to timelag xyz data file')
    parser.add_argument('--current_distance', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/current-distance.pt", help='Path to current distance data file')
    parser.add_argument('--current_label', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/current-label.pt", help='Path to current distance data file')
    parser.add_argument('--n_epochs', type=int, default="1000", help='Number of epochs to train')
    parser.add_argument('--n_epochs_finetune', type=int, default="500", help='Number of epochs to train')
    parser.add_argument('--n_batch', type=int, default="256", help='Data batch size')
    parser.add_argument('--sigma', type=float, default="0.00", help='Sigma value for CNF')
    parser.add_argument('--hidden_dim', type=int, default="64", help='hidden dimension of EGNN')
    parser.add_argument('--cv_dimension', type=int, default="2", help='cv dimension')
    parser.add_argument('--warmup', type=int, default="50", help='Number of epochs for scheduler warmup')
    parser.add_argument('--type', type=str, default="cv-condition", help='training type')
    parser.add_argument('--cfg_p', type=float, default=0.2, help='Threshold for CFG')
    parser.add_argument('--tags', nargs='*', help='Tags for Wandb')
    
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
save_dir = f"res/{args.date}/model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(f"Model will be saved to {save_dir}")
json.dump(
    vars(args),
    open(save_dir + "/args.json", "w"),
)

if args.type == "cv-condition":
    encoder_layers = [45, 30, 30, args.cv_dimension]
    cv_dimension = encoder_layers[-1]
    tbgcv = TBGCV(encoder_layers=encoder_layers).cuda()
    tbgcv.eval()
elif args.type == "label":
    cv_dimension = args.cv_dimension
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
data_current_xyz = torch.load(args.current_xyz).cuda().reshape(-1, 22 * 3)
data_current_distance = torch.load(args.current_distance).cuda()
data_timelag_xyz = torch.load(args.timelag_xyz).cuda()
batch_iter = IndexBatchIterator(len(data_current_xyz), args.n_batch)
if args.type == "cv-condition":
    wandb.watch([tbgcv, net_dynamics], log="all", log_freq=100)
else:
    wandb.watch(net_dynamics, log="all", log_freq=100)
optim_tbg = torch.optim.AdamW(flow.parameters(), lr=1e-6, weight_decay=1e-3)
scheduler_tbg = CosineAnnealingWarmUpRestarts(optim_tbg, T_0=args.n_epochs + args.n_epochs_finetune, T_up=args.warmup, eta_max=5e-4, gamma=0.5)

optim_cv = torch.optim.AdamW(tbgcv.parameters(), lr=1e-6, weight_decay=1e-3)
scheduler_cv = CosineAnnealingWarmUpRestarts(optim_cv, T_0=args.n_epochs_finetune, T_up=args.warmup, eta_max=5e-4, gamma=0.5)

epoch_loss = torch.tensor(0.0).cuda()
pbar = tqdm(
    range(args.n_epochs + args.n_epochs_finetune),
    desc = f"Loss: {epoch_loss:.4f}"
)
for epoch in pbar:
    loss_list = []
    if epoch > args.n_epochs:
        tbgcv.train()
    
    for it, idx in enumerate(batch_iter):
        optim_tbg.zero_grad()

        # Load data
        x1_current = data_current_xyz[idx]
        batchsize = x1_current.shape[0]
        
        # type
        if args.type == "cv-condition":
            x1 = data_timelag_xyz[idx]
            if epoch > args.n_epochs:
                x1_distance = data_current_distance[idx]
                cv_condition = tbgcv(x1_distance)
            else:
                cv_condition = torch.zeros(batchsize, cv_dimension).cuda()
        else:
            raise ValueError(f"Unknown training type {args.type}")

        # calculate regression loss
        t = torch.rand(batchsize, 1).cuda()
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
        
        # Step
        optim_tbg.step()
        scheduler_tbg.step(epoch + it / len(batch_iter))
        if epoch > args.n_epochs:
            optim_cv.step()
            scheduler_cv.step(epoch - args.n_epochs + it / len(batch_iter))
    epoch_loss = np.mean(loss_list)
    pbar.set_description(f"Loss: {epoch_loss:.4f}")
    
    wandb.log({
        "loss": epoch_loss,
        "lr/tbg": scheduler_tbg.get_last_lr()[0],
        "lr/cv": scheduler_cv.get_last_lr()[0],
        "model/cv": tbgcv.training
    }, step=epoch)
    
    if epoch % 400 == 0 and epoch != 0:
        print(f"Epoch {epoch}")
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim_tbg.state_dict(),
                "epoch": epoch,
            },
            save_dir + f"/tbg-{epoch}.pt",
        )
        if args.type not in ["repro", "label"]:    
            torch.save(
                tbgcv.state_dict(),
                save_dir + f"/mlcv-{epoch}.pt",   
            )
        
print(f">> Final epoch {epoch}")
torch.save(
    {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optim_tbg.state_dict(),
        "epoch": epoch,
    },
    save_dir + f"/tbg-final.pt",
)
wandb.save(save_dir + f"/tbg-final.pt")

if args.type not in ["repro", "label"]:
    torch.save(
        tbgcv.state_dict(),
        save_dir + f"/mlcv-final.pt",   
    )
    wandb.save(save_dir + f"/mlcv-final.pt")
print(f"Model saved to {save_dir}")

wandb.finish()