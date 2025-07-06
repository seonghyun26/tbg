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

from tbg.utils import create_adjacency_list, find_chirality_centers, compute_chirality_sign, check_symmetry_change
from tbg.cv import TBGCV


PHI_ANGLE = [4, 6, 8, 14]
PSI_ANGLE = [6, 8, 14, 16]
ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
FONTSIZE = 28
FONTSIZE_SMALL = 20


def align_topology(sample, reference, scaling=1):
    sample = sample.reshape(-1, 3)
    all_dists = scipy.spatial.distance.cdist(sample, sample)
    adj_list_computed = create_adjacency_list(all_dists/scaling, atom_types)
    G_reference = nx.Graph(reference)
    G_sample = nx.Graph(adj_list_computed)
    # not same number of nodes
    if len(G_sample.nodes) != len(G_reference.nodes):
        return sample, False
    for i, atom_type in enumerate(atom_types):
        G_reference.nodes[i]['type']=atom_type
        G_sample.nodes[i]['type']=atom_type
        
    nm = iso.categorical_node_match("type", -1)
    GM = isomorphism.GraphMatcher(G_reference, G_sample, node_match=nm)
    is_isomorphic = GM.is_isomorphic()
    
    GM.mapping
    initial_idx = list(GM.mapping.keys())
    final_idx = list(GM.mapping.values())
    sample[initial_idx] = sample[final_idx]
    return sample, is_isomorphic

def parse_args():
    parser = argparse.ArgumentParser(description='Sampling from TBG model')
    parser.add_argument('--date', type=str, default= "debug", help='Date for the experiment')
    parser.add_argument('--type', type=str, default="cv-condition", help='training type')
    parser.add_argument('--state', type=str, default= "c5", help='Conditioning state')
    parser.add_argument('--scaling', type=float, default= "1", help='Scaling on data')
    parser.add_argument('--cv_dimension', type=int, default="2", help='cv dimension')
    parser.add_argument('--tags', nargs='*', default=["evaulation"], help='Tags for Wandb')
    
    return parser.parse_args()

args = parse_args()


n_particles = 22
n_dimensions = 3
scaling = args.scaling
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
else:
    raise ValueError(f"Invalid type: {args.type}")

tbgcv_checkpoint = torch.load(f"./res/{args.date}/model/mlcv-final.pt")
tbgcv.load_state_dict(tbgcv_checkpoint)
tbgcv.eval()


# Load dataset and samples for evaluation
dataset = AImplicitUnconstrained(root=os.getcwd()+"/../tbg/", read=True, download=False)
data = dataset.xyz
target = dataset.get_energy_model()
latent_np = torch.load(f"./res/{args.date}/result/latent-{args.state}.pt").cpu().detach().numpy()
samples_np = torch.load(f"./res/{args.date}/result/samples-{args.state}.pt").cpu().detach().numpy()
dlogp_np = torch.load(f"./res/{args.date}/result/dlogp-{args.state}.pt").cpu().detach().numpy()


print(">> Aligning samples")
aligned_samples = []
aligned_idxs = []
atom_dict = {"C": 0, "H":1, "N":2, "O":3}
topology = md.load(f"data/AD2/c5.pdb").topology
atom_types = []
for atom_name in topology.atoms:
    atom_types.append(atom_name.name[0])
atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
adj_list = torch.from_numpy(np.array([(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32))
pbar = tqdm(
    samples_np.reshape(-1,dim//3, 3),
    desc = f"Aligned samples 0"
)
for i, sample in enumerate(pbar):   
    aligned_sample, is_isomorphic = align_topology(sample, as_numpy(adj_list).tolist(), scaling = scaling)
    if is_isomorphic:
        aligned_samples.append(aligned_sample)
        aligned_idxs.append(i)
    pbar.set_description(f"Aligned samples {len(aligned_samples)}")
aligned_samples = np.array(aligned_samples)
aligned_samples.shape
print(f"Correct configuration rate {len(aligned_samples)/len(samples_np)}")
wandb.log({"correct_configuration_rate": len(aligned_samples)/len(samples_np)})
if len(aligned_samples) == 0:
    print("No samples were aligned correctly.")
    aligned_samples = samples_np


# Process chirality
print(">> Processing chirality")
traj_samples = md.Trajectory(aligned_samples/scaling, topology=topology)
model_samples = torch.from_numpy(traj_samples.xyz)
chirality_centers = find_chirality_centers(adj_list, atom_types)
reference_signs = compute_chirality_sign(torch.from_numpy(data.reshape(-1, dim//3, 3))[[1]], chirality_centers)
symmetry_change = check_symmetry_change(model_samples, chirality_centers, reference_signs)
model_samples[symmetry_change] *=-1
symmetry_change = check_symmetry_change(model_samples, chirality_centers, reference_signs)
print(f"Correct symmetry rate {(~symmetry_change).sum()/len(model_samples)}")
wandb.log({"correct_symmetry_rate": (~symmetry_change).sum()/len(model_samples)})
traj_samples = md.Trajectory(as_numpy(model_samples)[~symmetry_change], topology=topology)
phis = md.compute_phi(traj_samples)[1].flatten()
psis = md.compute_psi(traj_samples)[1].flatten()


# Plot Ramachandran plot
print(">> Plotting Ramachandran plot")
plot_range = [-np.pi, np.pi]
plt.clf()
fig, ax = plt.subplots(figsize=(6, 6))
h, x_bins, y_bins, im = ax.hist2d(
    phis, psis, 100,
    norm=LogNorm(),
    range=[plot_range,plot_range],
    rasterized=True
)
ax.scatter(-2.49, 2.67, edgecolors="black", c="w", zorder=101, s=100)
ax.scatter(1.02, -0.70, edgecolors="black", c="w", zorder=101, s=300, marker="*")
ax.margins(0) 
ax.tick_params(
    left = False,
    right = False ,
    labelleft = True , 
    labelbottom = True,
    bottom = False
) 
ax.set_xlabel(r"$\phi$", fontsize=FONTSIZE)
ax.set_ylabel(r"$\psi$", fontsize=FONTSIZE)
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
plt.savefig(f"{save_dir}/{args.state}-ram.png")
wandb.log({"ramachandran_plot": wandb.Image(fig)})
print(f"Ramachandran plot saved at {save_dir}/{args.state}-ram.png")
plt.close()


# Energy evaluation
print("Evaluating energy")
# classical_model_energies = as_numpy(target.energy(model_samples.reshape(-1, dim)[~symmetry_change]))
# classical_target_energies = as_numpy(target.energy(torch.from_numpy(dataset.xyz[::100]).reshape(-1, dim)))
# prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
# idxs = np.array(aligned_idxs)[~symmetry_change]
# log_w_np = -classical_model_energies + as_numpy(prior.energy(torch.from_numpy(latent_np[idxs]).cuda())) + dlogp_np.reshape(-1,1)[idxs]
# np.save(f"{save_dir}/{args.state}-classical_target_energies.npy", classical_target_energies)
# np.save(f"{save_dir}/{args.state}-classical_model_energies.npy", classical_model_energies)
# print(">>Plotting energy distribution")

# fig = plt.figure(figsize=(16, 9))
# plt.hist(classical_target_energies, bins=100, alpha=0.5, range=(-50,100), density=True, label="MD")
# plt.hist(classical_model_energies, bins=100,alpha=0.5, range=(-50,100), density=True, label="BG")
# plt.hist(classical_model_energies, bins=100,alpha=0.5, range=(-50,100), density=True, label="BG weighted", weights=np.exp(log_w_np))
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(fontsize=20)
# plt.xlabel("Energy in kbT", fontsize=20) 
# plt.title("Classical - Energy distribution", fontsize=25)
# plt.savefig(f"{save_dir}/{args.state}-energy_distribution.png")
# plt.close()
# print(f"Energy distribution saved at {save_dir}/{args.state}-energy_distribution.png")
# wandb.log({"energy_distribution": wandb.Image(fig)})

wandb.finish()
exit(0)