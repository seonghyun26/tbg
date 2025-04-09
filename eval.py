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
    parser.add_argument('--file_name', type=str, default= "tbg-v1", help='Path to saved file')
    parser.add_argument('--state', type=str, default= "c5", help='Conditioning state')
    parser.add_argument('--topology', type=str, default= "c5-tbg", help='State file name for topology')
    parser.add_argument('--scaling', type=int, default= "1", help='Scaling on data')
    return parser.parse_args()

args = parse_args()


n_particles = 22
n_dimensions = 3
scaling = args.scaling
dim = n_particles * n_dimensions

latent_np = torch.load(f"./result_data/{args.file_name}/latent-{args.state}.pt").cpu().detach().numpy()
samples_np = torch.load(f"./result_data/{args.file_name}/samples-{args.state}.pt").cpu().detach().numpy()
dlogp_np = torch.load(f"./result_data/{args.file_name}/dlogp-{args.state}.pt").cpu().detach().numpy()
topology_file = f"data/AD2/{args.topology}.pdb"
ad2_topology = md.load(topology_file).topology


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


# aligned_samples = []
# aligned_idxs = []
# for i, sample in enumerate(tqdm(
#     samples_np.reshape(-1,dim//3, 3),
#     desc = f"Computing samples from npy file"
# )):
#     aligned_sample, is_isomorphic = align_topology(sample, as_numpy(adj_list).tolist())
#     if is_isomorphic:
#         aligned_samples.append(aligned_sample)
#         aligned_idxs.append(i)
# aligned_samples = np.array(aligned_samples)
# aligned_samples.shape
# print(f"Correct configuration rate {len(aligned_samples)/len(samples_np)}")
# print(f"Aligned samples shape: {aligned_samples.shape}")


aligned_samples = samples_np.reshape(-1, 22, 3)
print(f"Aligned samples shape: {aligned_samples.shape}")



traj_samples = md.Trajectory(aligned_samples/scaling, topology=topology)
phis = md.compute_phi(traj_samples)[1].flatten()
psis = md.compute_psi(traj_samples)[1].flatten()
sample_for_chirality = md.load(topology_file).xyz
chirality_sample_c5 = torch.from_numpy(sample_for_chirality)
model_samples = torch.from_numpy(traj_samples.xyz)
chirality_centers = find_chirality_centers(adj_list, atom_types)
reference_signs = compute_chirality_sign(chirality_sample_c5, chirality_centers)
symmetry_change = check_symmetry_change(model_samples, chirality_centers, reference_signs)
model_samples[symmetry_change] *=-1
symmetry_change = check_symmetry_change(model_samples, chirality_centers, reference_signs)
print(f"Correct symmetry rate {(~symmetry_change).sum()/len(model_samples)}")


# traj_samples = md.Trajectory(as_numpy(model_samples)[~symmetry_change], topology=topology)
traj_samples = md.Trajectory(as_numpy(model_samples), topology=topology)
print(traj_samples.xyz.shape)
phis = md.compute_phi(traj_samples)[1].flatten()
psis = md.compute_psi(traj_samples)[1].flatten()
# print(phis)
# print(psis)

# # Plot Ramachandran plot
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
fig.savefig(f'./result_data/{args.file_name}/ram-{args.state}.png')  # Save the figure to a file
print(f"Image saved at {f'./result_data/{args.file_name}/ram-{args.state}.png'}")

# Plot scatter plot
fig, ax = plt.subplots(figsize=(11, 9))
ax.scatter(phis, psis, s=10)
ax.set_xlabel(r"$\varphi$", fontsize=45)
ax.set_title("Boltzmann Generator", fontsize=45)
ax.xaxis.set_tick_params(labelsize=25)
ax.yaxis.set_tick_params(labelsize=25)
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)
fig.show()
fig.savefig(f'./result_data/{args.file_name}/scatter-{args.state}.png')  # Save the figure to a file
print(f"Image saved at {f'./result_data/{args.file_name}/scatter-{args.state}.png'}")


# Plot energy distribution
# from bgmol.datasets import AImplicitUnconstrained
# dataset = AImplicitUnconstrained(read=True, download=False)
# target = dataset.get_energy_model()
# print("Computing energy...")
# classical_model_energies = target.energy(model_samples.reshape(-1, dim)).cpu().detach().numpy()
# print(classical_model_energies.shape)
# print(classical_model_energies.max())
# print(classical_model_energies.min())
# # classical_model_energies = np.log10(classical_model_energies)
# prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
# log_w_np = -classical_model_energies \
#     + (prior.energy(torch.from_numpy(latent_np).cuda())).cpu().detach().numpy() \
#     + dlogp_np.reshape(-1,1)

# print("Plotting energy distribution...")
# plt.figure(figsize=(16,9))
# # plt.hist(classical_target_energies, bins=100, alpha=0.5, density=True, label="MD");
# plt.hist(classical_model_energies, bins=100,alpha=0.5, range=(-50, 100), density=True, label="BG");
# # plt.hist(classical_model_energies, bins=100,alpha=0.5, range=(-50, 100), density=True, label="BG weighted", weights=np.exp(log_w_np));
# plt.legend(fontsize=25)
# plt.xlabel("Energy in kbT", fontsize=25)  
# plt.yticks(fontsize=20);
# plt.title("Classical - Energy distribution", fontsize=45)
# plt.show()
# plt.savefig(f'./result_data/{args.file_name}-energy.png')  # Save the figure to a file
# plt.close()
# print("Done!")