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
ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]


def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--date', type=str, default= "debug", help='Date for the experiment')
    parser.add_argument('--current_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/current-xyz.pt", help='Path to current xyz data file')
    parser.add_argument('--timelag_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/timelag-xyz.pt", help='Path to timelag xyz data file')
    parser.add_argument('--current_distance', type=str, default= "../../simulation/dataset/alanine/300.0/tbg-10n/current-distance.pt", help='Path to current distance data file')
    parser.add_argument('--n_epochs', type=int, default="1000", help='Number of epochs to train')
    parser.add_argument('--n_batch', type=int, default="256", help='Data batch size')
    parser.add_argument('--sigma', type=float, default="0.00", help='Sigma value for CNF')
    parser.add_argument('--hidden_dim', type=int, default="64", help='hidden dimension of EGNN')
    parser.add_argument('--cv_dimension', type=int, default="2", help='cv dimension')
    parser.add_argument('--warmup', type=int, default="50", help='Number of epochs for scheduler warmup')
    parser.add_argument('--type', type=str, default="cv-condition", help='training type')
    parser.add_argument('--normalization', type=bool, default=False, help='Normalization for data')
    parser.add_argument('--ac_loss_lambda', type=float, default=0.3, help='AC loss lambda')
    parser.add_argument('--tags', nargs='*', help='Tags for Wandb')
    
    return parser.parse_args()

args = parse_args()
n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
atom_types = np.arange(n_particles)
atom_types[[0, 2, 3]] = 2
atom_types[1] = 0
atom_types[[11, 12, 13]] = 12
atom_types[[19, 20, 21]] = 20
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))

# Logging
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


# Setup mlcv model
if args.type in ["cv-condition", "cfg"]:
    encoder_layers = [45, 30, 30, args.cv_dimension]
elif args.type in ["cv-condition-xyz", "cv-condition-xyz-ac"]:
    encoder_layers = [30, 100, 100, args.cv_dimension]
elif args.type in ["cv-condition-xyzhad"]:
    encoder_layers = [75, 100, 100, args.cv_dimension]
tbgcv = TBGCV(
    encoder_layers = encoder_layers,
    options = {
        "encoder": {
            "activation": "tanh",
            "dropout": [0.5, 0.5, 0.5]
        },
        "norm_in": {
        },
    },
).cuda()
tbgcv.train()

# Setup flow model
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
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
    cv_dimension=args.cv_dimension,
    mode="egnn_dynamics",
    agg="sum",
)
bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator
)
flow = DiffEqFlow(dynamics=bb_dynamics)
if args.type in ["cv-condition", "cfg"]:
    wandb.watch([tbgcv, net_dynamics], log="parameters", log_freq=100)
else:
    wandb.watch(net_dynamics, log="parameters", log_freq=100)

# Load dataset
data_current_distance = torch.load(args.current_distance).cuda()
data_current_xyz = torch.load(args.current_xyz).cuda().reshape(-1, 22, 3)[:, ALANINE_HEAVY_ATOM_IDX, :].reshape(-1, 10 * 3)
data_timelag_xyz = torch.load(args.timelag_xyz).cuda().reshape(-1, 22 * 3)
batch_iter = IndexBatchIterator(len(data_timelag_xyz), args.n_batch)
if args.normalization:
    print("Normalizing data")
    parent_dir = os.path.dirname(args.current_xyz)
    cfg_path = os.path.join(parent_dir, "cfg_list.json")
    with open(cfg_path, "r") as f:
        cfg_list = json.load(f)
    data_stat = cfg_list[-1]
    print(f"Dataset statistics: {data_stat}")
    data_current_distance = (data_current_distance - data_stat["current_distance_mean"]) / data_stat["current_distance_std"]
    data_current_xyz = (data_current_xyz - data_stat["current_xyz_mean"]) / data_stat["current_xyz_std"]
    data_timelag_xyz = (data_timelag_xyz - data_stat["time_lagged_xyz_mean"]) / data_stat["time_lagged_xyz_std"]
else:
    print("No normalization applied")

# Set optimizer, scheduler
optim = torch.optim.AdamW(
    list(flow.parameters()) + list(tbgcv.parameters()),
    lr=1e-6,
    weight_decay=1e-3
)
scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=args.n_epochs, T_up=args.warmup, eta_max=5e-4, gamma=0.5)

epoch_loss = torch.tensor(0.0).cuda()
pbar = tqdm(range(args.n_epochs), desc = f"Loss: {epoch_loss:.4f}",)
for epoch in pbar:
    loss_list = []
    loss_ac_list = []
    
    for it, idx in enumerate(batch_iter):
        optim.zero_grad()
        batch_size = data_timelag_xyz[idx].shape[0]
        
        # type
        if args.type == "cv-condition":
            x1 = data_timelag_xyz[idx]
            x1_distance = data_current_distance[idx]
            cv_condition = tbgcv(x1_distance)

        elif args.type == "cv-condition-xyz":
            x1 = data_timelag_xyz[idx]
            x1_xyz = data_current_xyz[idx]
            cv_condition = tbgcv(x1_xyz)
            
        elif args.type == "cv-condition-xyz-ac":
            x1 = data_timelag_xyz[idx]
            x1_xyz = data_current_xyz[idx]
            timelag_xyz = data_timelag_xyz[idx]
            cv_condition = tbgcv(x1_xyz)
            
            # autocorrealtion loss
            s_t = tbgcv(x1_xyz)
            heavy_atom_xyz = timelag_xyz.reshape(batch_size, -1, 3)[:, ALANINE_HEAVY_ATOM_IDX].reshape(-1, 10 * 3)
            s_t_tau = tbgcv(heavy_atom_xyz)
            s_t_centered = s_t - s_t.mean(dim=0).repeat(s_t.shape[0], 1)
            s_t_tau_centered = s_t_tau - s_t_tau.mean(dim=0).repeat(s_t_tau.shape[0], 1)
            loss_ac = - (s_t_centered @ s_t_tau_centered.T)[torch.eye(s_t.shape[0], dtype=torch.bool, device = s_t.device)].mean()
            loss_ac = loss_ac / (s_t.std(dim=0).T @ s_t_tau.std(dim=0))
            loss_ac_list.append(loss_ac.item())

        elif args.type == "cv-condition-xyzhad":
            x1 = data_timelag_xyz[idx]
            x1_xyz = data_current_xyz[idx]
            x1_had = data_current_distance[idx]
            x1_input = torch.cat([x1_xyz, x1_had], dim=1)
            cv_condition = tbgcv(x1_input)

        else:
            raise ValueError(f"Unknown training type {args.type}")

        # Compute flow
        t = torch.rand(batch_size, 1).cuda()
        x0 = prior.sample(batch_size)
        mu_t = x0 * (1 - t) + x1 * t
        noise = prior.sample(batch_size)
        x = mu_t + args.sigma * noise
        ut = x1 - x0
        
        # Compute regression loss
        vt = flow._dynamics._dynamics._dynamics_function(t, x, cv_condition)
        if args.type == "cv-condition-xyz-ac":
            loss = torch.mean((vt - ut) ** 2) + args.ac_loss_lambda * loss_ac
        else:
            loss = torch.mean((vt - ut) ** 2)
        loss_list.append(loss.item())
        loss.backward()
        
        # Step
        optim.step()
        scheduler.step(epoch + int(it / len(batch_iter)))
        # optim_tbg.step()
        # scheduler_tbg.step(epoch + it / len(batch_iter))
        # optim_cv.step()
        # scheduler_cv.step(epoch + it / len(batch_iter))
    epoch_loss = np.mean(loss_list)
    loss_ac = np.mean(loss_ac_list)
    pbar.set_description(f"Loss: {epoch_loss:.4f}")
    
    wandb.log({
        "loss": epoch_loss,
        "loss/fm": epoch_loss - args.ac_loss_lambda * loss_ac,
        "loss/ac": loss_ac,
        "lr": scheduler.get_last_lr()[0],
        # "lr/tbg": scheduler_tbg.get_last_lr()[0],
        # "lr/cv": scheduler_cv.get_last_lr()[0],
    }, step=epoch)
    
    if epoch == 500:
        print(f"Epoch {epoch}")
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            save_dir + f"/tbg-{epoch}.pt",
        )
        if args.type not in ["repro", "label"]:    
            torch.save(
                tbgcv.state_dict(),
                save_dir + f"/mlcv-{epoch}.pt",   
            )
        
print(f">> Training finished")
torch.save(
    {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
    },
    save_dir + f"/tbg-final.pt",
)


torch.save(
    tbgcv.state_dict(),
    save_dir + f"/{args.date}.pt",   
)
torch.save(
    tbgcv.state_dict(),
    save_dir + f"/mlcv-final.pt",   
)
# dummy_trainer = Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)
# tbgcv.trainer = dummy_trainer
# random_input = torch.rand(1, x1_distance.shape[1])
# traced_script_module = torch.jit.trace(tbgcv, random_input)
# traced_script_module.save(f"{save_dir}/mlcv-final-jit.pt")

print(f"Model saved to {save_dir}")
wandb.finish()