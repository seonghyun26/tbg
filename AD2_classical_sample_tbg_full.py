import tqdm
import torch
import wandb
import numpy as np

import argparse

from bgflow.utils import as_numpy
from bgflow import DiffEqFlow, BoltzmannGeneratorCV, MeanFreeNormalDistribution
from bgflow import BlackBoxDynamics, BruteForceEstimator
from tbg.models2 import EGNN_dynamics_AD2_cat
from tbg.modelwithcv import EGNN_AD2_CV
from bgflow import BlackBoxDynamics, BruteForceEstimator


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from TBG')
    parser.add_argument('--data_xyz_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg3.pt", help='Path to xyz data file')
    parser.add_argument('--data_distance_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg3.pt", help='Path to distance data file')
    parser.add_argument('--filename', type=str, default= "tbgcv", help='checkpoint name')
    parser.add_argument('--hidden_dim', type=int, default="64", help='hidden dimension of EGNN')
    parser.add_argument('--state', type=str, default="c5", help='one state condition for sampling')
    parser.add_argument('--n_samples', type=int, default= 400, help='number of samples')
    parser.add_argument('--n_sample_batches', type=int, default= 500, help='number of samples batch')
    parser.add_argument('--tags', nargs='*', help='Tags for Wandb')
    
    return parser.parse_args()

args = parse_args()


# atom types for backbone
n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
scaling = 10

wandb.init(
    project="tbg",
    entity="eddy26",
    tags=["original"] + args.tags,
)

atom_types = np.arange(22)
# atom_types[[0, 2, 3]] = 0
# atom_types[1] = 2
atom_types[[1, 2, 3]] = 2
atom_types[[11, 12, 13]] = 12
atom_types[[19, 20, 21]] = 20
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)

brute_force_estimator = BruteForceEstimator()
# net_dynamics = EGNN_AD2_CV(
net_dynamics = EGNN_dynamics_AD2_cat(
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

            dxs = dynamics(t, torch.cat(x, dim=1))

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            divergence = 0
            for i in range(xs.size(1)):
                divergence += torch.autograd.grad(
                    dxs[:, [i]], x[i], torch.ones_like(dxs[:, [i]]), retain_graph=True
                )[0]

        return dxs, -divergence.view(-1, 1)


brute_force_estimator_fast = BruteForceEstimatorFast()
# use OTD in the evaluation process
bb_dynamics._divergence_estimator = brute_force_estimator_fast
bg.flow._integrator_atol = 1e-4
bg.flow._integrator_rtol = 1e-4
flow._use_checkpoints = False
flow._kwargs = {}


filename = "FM-AD2-train-repro-custom-data"
# filename = "tbg-fixed6.pt"
PATH_last = f"models/{filename}"
checkpoint = torch.load(PATH_last)
flow.load_state_dict(checkpoint["model_state_dict"])

# n_samples = 400
# n_sample_batches = 500
n_samples = args.n_samples
n_sample_batches = args.n_sample_batches
latent_np = np.empty(shape=(0))
samples_np = np.empty(shape=(0))
dlogp_np = np.empty(shape=(0))

print(f"Start sampling with {filename}")
for i in tqdm.tqdm(range(n_sample_batches)):
    with torch.no_grad():
        samples, latent, dlogp = bg.sample(n_samples, cv_condition=None, with_latent=True, with_dlogp=True)
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
