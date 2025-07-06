import os
import wandb
import torch
import scipy
import numpy as np
import argparse
from tqdm import tqdm

import networkx.algorithms.isomorphism as iso
import networkx as nx
from networkx import isomorphism
from bgmol.datasets import AImplicitUnconstrained
from bgflow.utils import as_numpy
from bgflow import MeanFreeNormalDistribution

import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import RegularPolygon


from tbg.utils import create_adjacency_list, find_chirality_centers, compute_chirality_sign, check_symmetry_change
from tbg.cv import TBGCV, PostProcess
from mlcolvar.core.transform import Statistics



PHI_ANGLE = [4, 6, 8, 14]
PSI_ANGLE = [6, 8, 14, 16]
ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]


def parse_args():
    parser = argparse.ArgumentParser(description='Sampling from TBG model')
    parser.add_argument('--date', type=str, default= "debug", help='Date for the experiment')
    parser.add_argument('--type', type=str, default="cv-condition", help='training type')
    parser.add_argument('--cv_dimension', type=int, default="2", help='cv dimension')
    parser.add_argument('--tags', nargs='*', default=["evaulation"], help='Tags for Wandb')
    
    return parser.parse_args()

args = parse_args()


n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
save_dir = f"./res/{args.date}"

wandb.init(
    project="tbg",
    entity="eddy26",
    config=vars(args),
    tags=["evaluation"] + args.tags,
)

if args.type in ["cv-condition", "cfg"]:
    encoder_layers = [45, 30, 30, args.cv_dimension]
    tbgcv = TBGCV(encoder_layers=encoder_layers).cuda()
    tbgcv.eval()
elif args.type in ["cv-condition-xyz", "cv-condition-xyz-ac"]:
    encoder_layers = [30, 100, 100, args.cv_dimension]
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
    tbgcv.eval()
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
    tbgcv.eval()
else:
    raise ValueError(f"Invalid type: {args.type}")

tbgcv_checkpoint = torch.load(f"./res/{args.date}/model/mlcv-final.pt")
tbgcv.load_state_dict(tbgcv_checkpoint)
tbgcv.eval()


# # Projection_dataset
print(">> Plotting MLCVs")
data_dir = f"../../simulation/data/alanine"
if args.type == "cv-condition":
    projection_dataset = torch.load(f"{data_dir}/uniform-heavy-atom-distance.pt").cuda()
elif args.type in ["cv-condition-xyz", "cv-condition-xyz-ac"]:
    projection_dataset = torch.load(f"{data_dir}/uniform-xyz-aligned.pt").cuda()
    projection_dataset = projection_dataset[:, ALANINE_HEAVY_ATOM_IDX].reshape(projection_dataset.shape[0], -1)
c5_state = torch.load(f"../../simulation/data/alanine/c5.pt")['xyz'].to(tbgcv.device)

cv = tbgcv(projection_dataset)
stats = Statistics(cv.cpu()).to_dict()
wandb.log({f"cv/{key}":item for key, item in stats.items()})
tbgcv.postprocessing = PostProcess(stats, tbgcv(c5_state[:, ALANINE_HEAVY_ATOM_IDX].reshape(1, -1))[0]).to(tbgcv.device)
cv_normalized = tbgcv.postprocessing(cv.to(tbgcv.device))
print(f">> Max CV: {cv_normalized.max()}, Min CV: {cv_normalized.min()}")
print(f">> C5 CV: {tbgcv(c5_state[:, ALANINE_HEAVY_ATOM_IDX].reshape(1, -1))}")
cv = tbgcv(projection_dataset).cpu().detach().numpy()


psi_list = np.load(f"{data_dir}/uniform-psi.npy")
phi_list = np.load(f"{data_dir}/uniform-phi.npy")
c5 = torch.load(f"{data_dir}/c5.pt")
c7ax = torch.load(f"{data_dir}/c7ax.pt")
phi_start, psi_start = c5["phi"], c5["psi"]
phi_goal, psi_goal = c7ax["phi"], c7ax["psi"]

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(1, 1, 1)
hb = ax.hexbin(
    phi_list, psi_list, C=cv[:, 0],  # data
    gridsize=30,                     # controls resolution
    reduce_C_function=np.mean,       # compute average per hexagon
    cmap='viridis',                  # colormap
    extent=[-np.pi, np.pi, -np.pi, np.pi]
)

verts = hb.get_offsets()     # shape (n_hexes, 2)
values = hb.get_array()      # color value (reduced C)

# Highlight bins with CV near 0 (you can adjust the threshold)
threshold = 0.1
mask = np.abs(values) < threshold

# Add outline for those bins
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
plt.savefig(f"{save_dir}/cv-hexplot-preview.png")
plt.close()
