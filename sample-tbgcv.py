import tqdm
import torch
import wandb
import numpy as np

import argparse

from bgflow.utils import as_numpy, IndexBatchIterator
from bgflow import DiffEqFlow, BoltzmannGeneratorCV, MeanFreeNormalDistribution
from bgflow import BlackBoxDynamics, BruteForceEstimator
from tbg.modelwithcv import EGNN_AD2_CV, EGNN_AD2_CFG
from bgflow import BlackBoxDynamics, BruteForceEstimator


ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]


def kabsch_rmsd(
    P: torch.Tensor,
    Q: torch.Tensor
) -> torch.Tensor:
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(-2, -1), q)
    U, S, Vt = torch.linalg.svd(H)
    
    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))  # B
    Vt[d < 0.0, -1] *= -1.0

    # Optimal rotation and translation
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))

    # Calculate RMSD
    P_aligned = torch.matmul(P, R.transpose(-2, -1)) + t
    rmsd = (P_aligned - Q).square().sum(-1).mean(-1).sqrt()
    
    return rmsd

def coordinate2distance(
    positions
):
    '''
        Compute pairwise distances between heavy atoms of alanine
        Args:
            positions (torch.Tensor): Positions of atoms (n_samples, *)
    '''
    
    distance_list = []
    
    for position in positions:
        position = position.reshape(-1, 3)
        heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
        num_heavy_atoms = len(heavy_atom_position)
        distance = []
        for i in range(num_heavy_atoms):
            for j in range(i+1, num_heavy_atoms):
                distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
        distance = torch.stack(distance)
        distance_list.append(distance)
    
    return torch.stack(distance_list)

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from TBG')
    parser.add_argument('--data_xyz_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg.pt", help='Path to xyz data file')
    parser.add_argument('--data_distance_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt", help='Path to distance data file')
    parser.add_argument('--data_type', type=str, default= "1n", help='data type')
    return parser.parse_args()

args = parse_args()

n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
scaling = 10
n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions

wandb.init(
    project="tbg",
    entity="eddy26",
)

atom_types = np.arange(22)
atom_types[[1, 2, 3]] = 2
atom_types[[19, 20, 21]] = 20
atom_types[[11, 12, 13]] = 12
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))

# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)

brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_AD2_CV(
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
bg = BoltzmannGeneratorCV(prior, flow, prior).cuda()


class BruteForceEstimatorFast(torch.nn.Module):
    """
    Exact bruteforce estimation of the divergence of a dynamics function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, dynamics, t, xs, cv_condition = None):

        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            x = [xs[:, [i]] for i in range(xs.size(1))]

            dxs = dynamics(t, torch.cat(x, dim=1), cv_condition=cv_condition)

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            divergence = 0
            for i in range(xs.size(1)):
                divergence += torch.autograd.grad(
                    dxs[:, [i]], x[i], torch.ones_like(dxs[:, [i]]), retain_graph=True
                )[0]

        return dxs, -divergence.view(-1, 1)


brute_force_estimator_fast = BruteForceEstimatorFast()
bb_dynamics._divergence_estimator = brute_force_estimator_fast
bg.flow._integrator_atol = 1e-4
bg.flow._integrator_rtol = 1e-4
flow._use_checkpoints = False
flow._kwargs = {}


filename = "tbg"
PATH_last = f"models/tbgcv/1n/{filename}.pt"
# PATH_last = f"models/tbgcv-both/10n/{filename}.pt"
checkpoint = torch.load(PATH_last)
flow.load_state_dict(checkpoint["model_state_dict"])


n_batch = 256
data_xyz_path = args.data_xyz_path
data_xyz = torch.load(data_xyz_path)
data_distance_path = args.data_distance_path
data_distance = torch.load(data_distance_path)
batch_iter = IndexBatchIterator(len(data_xyz), n_batch)

# n_samples = 400
# n_sample_batches = 500
n_samples = n_batch
n_sample_batches = 400
latent_np = np.empty(shape=(0))
samples_np = np.empty(shape=(0))
dlogp_np = np.empty(shape=(0))
print(f"Start sampling with {filename}")

# Condition CV on c5 state
# c5_path = f"../../simulation/data/alanine/c7ax.pt"
# c5_xyz = torch.load(c5_path)['xyz']
# c5_heavy_atom_distance = coordinate2distance(c5_xyz).cuda()
# c5_heavy_atom_distance = c5_heavy_atom_distance.repeat(n_samples, 1)
# for i in tqdm.tqdm(range(n_sample_batches)):
#     with torch.no_grad():
#         samples, latent, dlogp = bg.sample(n_samples, cv_condition=c5_heavy_atom_distance, with_latent=True, with_dlogp=True)
#         latent_np = np.append(latent_np, latent.detach().cpu().numpy())
#         samples_np = np.append(samples_np, samples.detach().cpu().numpy())
#         dlogp_np = np.append(dlogp_np, as_numpy(dlogp))


rmsd = []
heavy_atom_distance_avg = []
heavy_atom_distance_difference = []
mse = torch.nn.MSELoss()

for it, idx in enumerate(tqdm.tqdm(batch_iter, desc="Sampling from BG")):
    x1 = data_xyz[idx][:, 1].cuda()
    x1_distance = data_distance[idx][:, 0].cuda()
    heavy_atom_distance_avg.append(x1_distance.mean())
    n_samples = x1.shape[0]
    with torch.no_grad():
        samples, latent, dlogp = bg.sample(n_samples, cv_condition=x1_distance, with_latent=True, with_dlogp=True)
        latent_np = np.append(latent_np, latent.detach().cpu().numpy())
        samples_np = np.append(samples_np, samples.detach().cpu().numpy())
        dlogp_np = np.append(dlogp_np, as_numpy(dlogp))
    
    rmsd_list = []
    for i in range(n_samples):
        sample = samples[i].reshape(-1, 3)
        reference = x1[i].reshape(-1, 3)
        rmsd_list.append(kabsch_rmsd(reference, sample))
    
    rmsd.append(torch.stack(rmsd_list).mean())
    heavy_atom_distance_difference.append(mse(coordinate2distance(samples), x1_distance))


wandb.log({
    "rmsd": torch.stack(rmsd).mean(),
    "heavy_atom_distance_difference": torch.stack(heavy_atom_distance_difference).mean(),
    "heavy_atom_distance_avg": torch.stack(heavy_atom_distance_avg).mean(),
    "heavy_atom_distance_std": torch.stack(heavy_atom_distance_avg).std(),
})
print(f"RMSD: {torch.stack(rmsd).mean()}")
print(f"Heavy atom distance difference: {torch.stack(heavy_atom_distance_difference).mean()}")
print(f"Heavy atom distance avg: {torch.stack(heavy_atom_distance_avg).mean()}")
print(f"Heavy atom distance std: {torch.stack(heavy_atom_distance_avg).std()}")

latent_np = latent_np.reshape(-1, dim)
samples_np = samples_np.reshape(-1, dim)
np.savez(
    f"result_data/{filename}-v2",
    latent_np=latent_np,
    samples_np=samples_np,
    dlogp_np=dlogp_np,
)
print(f"Saved data at {filename}")