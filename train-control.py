import os
import json
import torch
import wandb
import numpy as np
import argparse

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, MeanFreeNormalDistribution
from tbg.modelwithcv import EGNN_CV_CONTROL
from tbg.cv import TBGCV, PostProcess
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm

# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.scheduler import CosineAnnealingWarmUpRestarts
from mlcolvar.core.transform import Statistics
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


PHI_ANGLE = [4, 6, 8, 14]
PSI_ANGLE = [6, 8, 14, 16]
ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]


def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--date', type=str, default= "debug", help='Date for the experiment')
    parser.add_argument('--current_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/10nano-v2/current-xyz-aligned.pt", help='Path to current xyz data file')
    parser.add_argument('--timelag_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/10nano-v2/timelag-xyz.pt", help='Path to timelag xyz data file')
    parser.add_argument('--ac_timelag_xyz', type=str, default= "../../simulation/dataset/alanine/300.0/10nano-v2/timelag-xyz-aligned.pt", help='Path to timelag xyz data file')
    parser.add_argument('--current_distance', type=str, default= "../../simulation/dataset/alanine/300.0/10nano-v2/current-distance.pt", help='Path to current distance data file')
    parser.add_argument('--n_epochs', type=int, default="1000", help='Number of epochs to train')
    parser.add_argument('--n_batch', type=int, default="256", help='Data batch size')
    parser.add_argument('--sigma', type=float, default="0.00", help='Sigma value for CNF')
    parser.add_argument('--dropout', type=float, default="0.5", help='Dropout value for MLCVs')
    parser.add_argument('--hidden_dim', type=int, default="64", help='hidden dimension of EGNN')
    parser.add_argument('--cv_dimension', type=int, default="2", help='cv dimension')
    parser.add_argument('--warmup', type=int, default="50", help='Number of epochs for scheduler warmup')
    parser.add_argument('--type', type=str, default="cv-condition", help='training type')
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
run = wandb.init(
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
if args.type in ["cv-condition-xyz"]:
    encoder_layers = [30, 100, 100, args.cv_dimension]
else:
    raise ValueError(f"Unknown training type {args.type}")

options = {
    "encoder": {
        "activation": "tanh",
        "dropout": [args.dropout, args.dropout, args.dropout]
    },
    "norm_in": {
    },
}
tbgcv = TBGCV(
    encoder_layers = encoder_layers,
    options = options
).cuda()
print(f"MLCV model: {tbgcv}")
tbgcv.train()
run.config.update(options)

# Setup flow model
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_CV_CONTROL(
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
path_to_pretrained_model = "./models/tbg-aldp-full.pt"
net_dynamics.load_pretrained(path_to_pretrained_model)
bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator
)
flow = DiffEqFlow(dynamics=bb_dynamics)
wandb.watch(net_dynamics, log="parameters", log_freq=100)


# Load dataset
data_current_distance = torch.load(args.current_distance).cuda()
data_current_xyz = torch.load(args.current_xyz).cuda().reshape(-1, 22, 3)[:, ALANINE_HEAVY_ATOM_IDX, :].reshape(-1, 10 * 3)
data_timelag_xyz = torch.load(args.timelag_xyz).cuda().reshape(-1, 22 * 3)
data_ac_timelag_xyz = torch.load(args.ac_timelag_xyz).cuda().reshape(-1, 22, 3)[:, ALANINE_HEAVY_ATOM_IDX, :].reshape(-1, 10 * 3)
batch_iter = IndexBatchIterator(len(data_timelag_xyz), args.n_batch)

# Set optimizer, scheduler
optim = torch.optim.AdamW(
    list(flow.parameters()) + list(tbgcv.parameters()),
    lr=1e-6,
    weight_decay=1e-3
)
scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=args.n_epochs, T_up=args.warmup, eta_max=5e-4)

epoch_loss = torch.tensor(0.0).cuda()
pbar = tqdm(range(args.n_epochs), desc = f"Loss: {epoch_loss:.4f}",)
for epoch in pbar:
    loss_list = []
    loss_ac_list = []
    
    for it, idx in enumerate(batch_iter):
        optim.zero_grad()
        batch_size = data_timelag_xyz[idx].shape[0]
        
        # type
        if args.type == "cv-condition-xyz":
            x1 = data_timelag_xyz[idx]
            x1_xyz = data_current_xyz[idx]
            cv_condition = tbgcv(x1_xyz)
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
        loss = torch.mean((vt - ut) ** 2)
        loss_list.append(loss.item())
        loss.backward()
        
        # Step
        optim.step()
        scheduler.step(epoch + int(it / len(batch_iter)))
    epoch_loss = np.mean(loss_list)
    pbar.set_description(f"Loss: {epoch_loss:.4f}")
    
    wandb.log({
        "loss": epoch_loss,
        "lr": scheduler.get_last_lr()[0],
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


# Load projection data
print(">> Plotting MLCVs")
tbgcv.eval()
data_dir = f"../../simulation/data/alanine"
if args.type in ["cv-condition-xyz"]:
    projection_dataset = torch.load(f"{data_dir}/uniform-xyz-aligned.pt").cuda()
    projection_dataset = projection_dataset[:, ALANINE_HEAVY_ATOM_IDX].reshape(projection_dataset.shape[0], -1)
else:
    raise ValueError(f"Unknown training type {args.type}")
c5_state = torch.load(f"../../simulation/data/alanine/c5.pt")['xyz'].to(tbgcv.device)
psi_list = np.load(f"{data_dir}/uniform-psi.npy")
phi_list = np.load(f"{data_dir}/uniform-phi.npy")
c5 = torch.load(f"{data_dir}/c5.pt")
c7ax = torch.load(f"{data_dir}/c7ax.pt")
phi_start, psi_start = c5["phi"], c5["psi"]
phi_goal, psi_goal = c7ax["phi"], c7ax["psi"]

# Compute MLCVs
cv = tbgcv(projection_dataset)
stats = Statistics(cv.cpu()).to_dict()
wandb.log({f"cv/{key}":item for key, item in stats.items()})
if args.type in ["cv-condition-xyz"]:
    tbgcv.postprocessing = PostProcess(stats, tbgcv(c5_state[:, ALANINE_HEAVY_ATOM_IDX].reshape(1, -1))[0]).to(tbgcv.device)
print(f">> Max CV: {cv.max()}, Min CV: {cv.min()}")


# Save figure
cv = tbgcv(projection_dataset).cpu().detach().numpy()
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(1, 1, 1)
hb = ax.hexbin(
    phi_list, psi_list, C=cv[:, 0],  # data
    gridsize=30,                     # controls resolution
    reduce_C_function=np.mean,       # compute average per hexagon
    cmap='viridis',                  # colormap
    extent=[-np.pi, np.pi, -np.pi, np.pi]
)
verts = hb.get_offsets()
values = hb.get_array()
threshold = 0.1
mask = np.abs(values) < threshold
for (x, y), val in zip(verts[mask], values[mask]):
    hex_patch = RegularPolygon(
        (x, y), numVertices=6,
        radius=hb.get_paths()[0].vertices[:, 0].max(),
        orientation=np.radians(30),
        facecolor='none', edgecolor='red', linewidth=1.2, zorder=102
    )
    ax.add_patch(hex_patch)
    
cbar = fig.colorbar(hb)
ax.scatter(phi_start, psi_start, edgecolors="black", c="w", zorder=101, s=100)
ax.scatter(phi_goal, psi_goal, edgecolors="black", c="w", zorder=101, s=300, marker="*")
print(f"MLCVs plot saved at {save_dir}/mlcv-hexplot.png")
wandb.log({"mlcv-hexplot-preview": wandb.Image(fig)})
plt.savefig(f"{save_dir}/mlcv-hexplot.png")
plt.close()


wandb.finish()