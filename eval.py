import os
import torch
import scipy
import numpy as np
import argparse


from bgmol.datasets import AImplicitUnconstrained
from bgflow.utils import (assert_numpy, distance_vectors, distances_from_vectors, 
                          remove_mean, IndexBatchIterator, LossReporter, as_numpy, compute_distances
)
from bgflow import (GaussianMCMCSampler, DiffEqFlow, BoltzmannGenerator, Energy, Sampler, 
                    MultiDoubleWellPotential, MeanFreeNormalDistribution, KernelDynamics)
from tbg.models2 import EGNN_dynamics_AD2, EGNN_dynamics_AD2_cat

import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tbg.utils import create_adjacency_list, find_chirality_centers, compute_chirality_sign, check_symmetry_change

from tqdm import tqdm

import networkx.algorithms.isomorphism as iso
import networkx as nx
from networkx import isomorphism


PHI_ANGLE = [4, 6, 8, 14]
PSI_ANGLE = [6, 8, 14, 16]
ALANINE_HEAVY_ATOM_IDX = [0, 4, 5, 6, 8, 10, 14, 15, 16, 18]


def compute_dihedral(positions):
    """http://stackoverflow.com/q/20305272/1128289"""
    def dihedral(p):
        if not isinstance(p, np.ndarray):
            p = p.numpy()
        b = p[:-1] - p[1:]
        b[0] *= -1
        v = np.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
        
        # Normalize vectors
        v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
        b1 = b[1] / np.linalg.norm(b[1])
        x = np.dot(v[0], v[1])
        m = np.cross(v[0], b1)
        y = np.dot(m, v[1])
        
        return np.arctan2(y, x)
    
    return np.array(list(map(dihedral, positions)))


def parse_args():
    parser = argparse.ArgumentParser(description='Sampling from TBG model')
    parser.add_argument('--date', type=str, default= "debug", help='Date for the experiment')
    parser.add_argument('--state', type=str, default= "c5", help='Conditioning state')
    parser.add_argument('--topology', type=str, default= "c5-tbg", help='State file name for topology')
    parser.add_argument('--scaling', type=float, default= "1", help='Scaling on data')
    return parser.parse_args()

args = parse_args()


n_particles = 22
n_dimensions = 3
scaling = args.scaling
dim = n_particles * n_dimensions
save_dir = f"./res/{args.date}"


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
    # True
    GM.mapping
    initial_idx = list(GM.mapping.keys())
    final_idx = list(GM.mapping.values())
    sample[initial_idx] = sample[final_idx]
    #print(is_isomorphic)
    return sample, is_isomorphic


# Load dataset and samples for evaluation
dataset = AImplicitUnconstrained(root=os.getcwd()+"/../tbg/", read=True, download=False)
data = dataset.xyz
target = dataset.get_energy_model()
data_energies = target.energy(torch.from_numpy(dataset.xyz[::10].reshape(-1,66)))
latent_np = torch.load(f"./res/{args.date}/result/latent-{args.state}.pt").cpu().detach().numpy()
samples_np = torch.load(f"./res/{args.date}/result/samples-{args.state}.pt").cpu().detach().numpy()
dlogp_np = torch.load(f"./res/{args.date}/result/dlogp-{args.state}.pt").cpu().detach().numpy()


print("Aligning samples")
aligned_samples = []
aligned_idxs = []
atom_dict = {"C": 0, "H":1, "N":2, "O":3}
# topology = dataset.system.mdtraj_topology
topology = md.load(f"data/AD2/c5-tbg.pdb").topology
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
if len(aligned_samples) == 0:
    raise ValueError("No samples were aligned correctly.")


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
traj_samples = md.Trajectory(as_numpy(model_samples)[~symmetry_change], topology=topology)
phis = md.compute_phi(traj_samples)[1].flatten()
psis = md.compute_psi(traj_samples)[1].flatten()


# Plot Ramachandran plot
print(">> Plotting Ramachandran plot")
plot_range = [-np.pi, np.pi]
fig, ax = plt.subplots(figsize=(11, 9))
h, x_bins, y_bins, im = ax.hist2d(phis, psis, 100, norm=LogNorm(), range=[plot_range,plot_range],rasterized=True)
ticks = np.array([np.exp(-6)*h.max(), np.exp(-4.0)*h.max(),np.exp(-2)*h.max(), h.max()])
ax.set_xlabel(r"$\varphi$", fontsize=45)
ax.set_title("Boltzmann Generator", fontsize=45)
ax.xaxis.set_tick_params(labelsize=25)
ax.yaxis.set_tick_params(labelsize=25)
cbar = fig.colorbar(im, ticks=ticks)
cbar.ax.set_yticklabels([6.0,4.0,2.0,0.0], fontsize=25)
cbar.ax.invert_yaxis()
cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
fig.savefig(f"{save_dir}/{args.state}-ram.png")
print(f"Ramachandran plot saved at {save_dir}/{args.state}-ram.png")



# Energy evaluation
print("Evaluating energy")
classical_model_energies = as_numpy(target.energy(model_samples.reshape(-1, dim)[~symmetry_change]))
classical_target_energies = as_numpy(target.energy(torch.from_numpy(dataset.xyz[::100]).reshape(-1, dim)))
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
idxs = np.array(aligned_idxs)[~symmetry_change]
log_w_np = -classical_model_energies + as_numpy(prior.energy(torch.from_numpy(latent_np[idxs]).cuda())) + dlogp_np.reshape(-1,1)[idxs]
np.save("classical_target_energies.npy", classical_target_energies)
print(">>Plotting energy distribution")
fig = plt.figure(figsize=(16, 9))
plt.hist(classical_target_energies, bins=100, alpha=0.5, range=(-50,100), density=True, label="MD")
plt.hist(classical_model_energies, bins=100,alpha=0.5, range=(-50,100), density=True, label="BG")
plt.hist(classical_model_energies, bins=100,alpha=0.5, range=(-50,100), density=True, label="BG weighted", weights=np.exp(log_w_np))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.xlabel("Energy in kbT", fontsize=20) 
plt.title("Classical - Energy distribution", fontsize=25)
fig.savefig(f"{save_dir}/{args.state}-energy_distribution.png")
print(f"Energy distribution saved at {save_dir}/{args.state}-energy_distribution.png")

exit(0)