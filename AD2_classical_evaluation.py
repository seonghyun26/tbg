import torch
import scipy
import numpy as np
import argparse


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

def parse_args():
    parser = argparse.ArgumentParser(description='Sampling from TBG model')
    parser.add_argument('--file_name', type=str, default= "sample_500", help='Path to saved file')
    return parser.parse_args()

args = parse_args()


n_particles = 22
n_dimensions = 3
scaling = 10
dim = n_particles * n_dimensions

npz_file = np.load(f"./result_data/{args.file_name}.npz")
latent_np=npz_file["latent_np"]
samples_np=npz_file["samples_np"]
dlogp_np=npz_file["dlogp_np"]
ad2_topology = md.load("data/AD2/c5.pdb").topology


def align_topology(sample, reference, scaling=scaling):
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


atom_dict = {"C": 0, "H":1, "N":2, "O":3}
topology = ad2_topology
atom_types = []
for atom_name in topology.atoms:
    atom_types.append(atom_name.name[0])
atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
adj_list = torch.from_numpy(np.array([(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32))


aligned_samples = []
aligned_idxs = []

for i, sample in enumerate(tqdm(
    samples_np.reshape(-1,dim//3, 3),
    desc = f"Computing samples from npy file"
)):
    aligned_sample, is_isomorphic = align_topology(sample, as_numpy(adj_list).tolist())
    if is_isomorphic:
        aligned_samples.append(aligned_sample)
        aligned_idxs.append(i)
aligned_samples = np.array(aligned_samples)
aligned_samples.shape
print(f"Correct configuration rate {len(aligned_samples)/len(samples_np)}")
print(f"Aligned samples shape: {aligned_samples.shape}")


aligned_samples = samples_np.reshape(-1, 22, 3)
print(f"Aligned samples shape: {aligned_samples.shape}")



traj_samples = md.Trajectory(aligned_samples/scaling, topology=topology)
phis = md.compute_phi(traj_samples)[1].flatten()
psis = md.compute_psi(traj_samples)[1].flatten()
sample_for_chirality = md.load("data/AD2/c5.pdb").xyz
chirality_sample_c5 = torch.from_numpy(sample_for_chirality)
model_samples = torch.from_numpy(traj_samples.xyz)
chirality_centers = find_chirality_centers(adj_list, atom_types)
reference_signs = compute_chirality_sign(chirality_sample_c5, chirality_centers)
symmetry_change = check_symmetry_change(model_samples, chirality_centers, reference_signs)
model_samples[symmetry_change] *=-1
symmetry_change = check_symmetry_change(model_samples, chirality_centers, reference_signs)
print(f"Correct symmetry rate {(~symmetry_change).sum()/len(model_samples)}")


traj_samples = md.Trajectory(as_numpy(model_samples)[~symmetry_change], topology=topology)

phis = md.compute_phi(traj_samples)[1].flatten()
psis = md.compute_psi(traj_samples)[1].flatten()


fig, ax = plt.subplots(figsize=(11, 9))
plot_range = [-np.pi, np.pi]

h, x_bins, y_bins, im = ax.hist2d(phis, psis, 100, norm=LogNorm(), range=[plot_range,plot_range],rasterized=True)
ticks = np.array([np.exp(-6)*h.max(), np.exp(-4.0)*h.max(),np.exp(-2)*h.max(), h.max()])
ax.set_xlabel(r"$\varphi$", fontsize=45)
ax.set_title("Boltzmann Generator", fontsize=45)
ax.xaxis.set_tick_params(labelsize=25)
ax.yaxis.set_tick_params(labelsize=25)
ax.yaxis.set_ticks([])

cbar = fig.colorbar(im, ticks=ticks)
cbar.ax.set_yticklabels([6.0,4.0,2.0,0.0], fontsize=25)
cbar.ax.invert_yaxis()
cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
fig.savefig(f'./result_data/{args.file_name}.png')  # Save the figure to a file



# Plot energy distribution
# classical_model_energies = as_numpy(target.energy(model_samples.reshape(-1, dim)[~symmetry_change]))
# prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
# idxs = np.array(aligned_idxs)[~symmetry_change]
# log_w_np = -classical_model_energies + as_numpy(prior.energy(torch.from_numpy(latent_np[idxs]).cuda())) + dlogp_np.reshape(-1,1)[idxs]