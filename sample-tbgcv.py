import tqdm
import torch
import wandb
import numpy as np

import argparse

from bgflow.utils import as_numpy
from bgflow import DiffEqFlow, BoltzmannGeneratorCV, MeanFreeNormalDistribution
from bgflow import BlackBoxDynamics, BruteForceEstimator
from tbg.modelwithcv import EGNN_AD2_CV
from bgflow import BlackBoxDynamics, BruteForceEstimator


ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]


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

def coordinate2distance(positions):
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
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--data_type', type=str, default= "1n", help='data type')
    parser.add_argument('--sample_epoch', type=int, default= "500", help='epoch interval for sampling')
    return parser.parse_args()

args = parse_args()


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
checkpoint = torch.load(PATH_last)
flow.load_state_dict(checkpoint["model_state_dict"])

# n_samples = 400
# n_sample_batches = 500
n_samples = 200
n_sample_batches = 400
latent_np = np.empty(shape=(0))
samples_np = np.empty(shape=(0))
dlogp_np = np.empty(shape=(0))
print(f"Start sampling with {filename}")

# Condition CV on c5 state
c5_path = f"../../simulation/data/alanine/c5.pt"
c5_xyz = torch.load(c5_path)['xyz']
c5_heavy_atom_distance = coordinate2distance(c5_xyz).cuda()
c5_heavy_atom_distance = c5_heavy_atom_distance.repeat(n_samples, 1)

for i in tqdm.tqdm(range(n_sample_batches)):
    with torch.no_grad():
        samples, latent, dlogp = bg.sample(n_samples, cv_condition=c5_heavy_atom_distance, with_latent=True, with_dlogp=True)
        latent_np = np.append(latent_np, latent.detach().cpu().numpy())
        samples_np = np.append(samples_np, samples.detach().cpu().numpy())
        dlogp_np = np.append(dlogp_np, as_numpy(dlogp))

    latent_np = latent_np.reshape(-1, dim)
    samples_np = samples_np.reshape(-1, dim)
    np.savez(
        f"result_data/{filename}",
        latent_np=latent_np,
        samples_np=samples_np,
        dlogp_np=dlogp_np,
    )
