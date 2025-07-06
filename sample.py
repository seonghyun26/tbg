import os
import tqdm
import torch
import wandb
import numpy as np
import time  # Add time module import

import argparse

from bgflow.utils import as_numpy, IndexBatchIterator
from bgflow import DiffEqFlow, BoltzmannGenerator, BoltzmannGeneratorCV, MeanFreeNormalDistribution
from bgflow import BlackBoxDynamics, BruteForceEstimator
# from tbg.eval import ALANINE_HEAVY_ATOM_IDX
from tbg.modelwithcv import EGNN_AD2_CV
from tbg.cv import TBGCV
from tbg.models2 import EGNN_dynamics_AD2_cat, EGNN_dynamics_AD2_cat_CV

from src.force import BruteForceEstimatorFast

ALANINE_HEAVY_ATOM_IDX= [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
ALANINE_HEAVY_ATOM_IDX_TBG = [0, 4, 5, 6, 8, 10, 14, 15, 16, 18]



def coordinate2distance(
    positions
): 
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

def kabsch(
    P: torch.Tensor,
    Q: torch.Tensor,
) -> torch.Tensor:
    '''
        Kabsch algorithm for aligning two sets of points
        Args:
            P (torch.Tensor): Current positions (N, 3)
            Q (torch.Tensor): Reference positions (N, 3)
        Returns:
            torch.Tensor: Aligned positions (N, 3)
    '''
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(P_centered.transpose(-2, -1), Q_centered)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))) 
    Vt[d < 0.0, -1] *= -1.0

    # Optimal rotation and translation
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))
    P_aligned = torch.matmul(P, R.transpose(-2, -1)) + t
    
    return P_aligned

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from TBG')
    parser.add_argument('--date', type=str, default= "debug", help='Date for the experiment')
    parser.add_argument('--hidden_dim', type=int, default="64", help='hidden dimension of EGNN')
    parser.add_argument('--cv_dimension', type=int, default="2", help='cv dimension')
    parser.add_argument('--guidance_scale', type=float, default="2.0", help='Guidance scale for CFG sampling')
    parser.add_argument('--state', type=str, default="c5", help='one state condition for sampling')
    parser.add_argument('--n_samples', type=int, default= 40, help='number of samples')
    parser.add_argument('--n_sample_batches', type=int, default= 20, help='number of samples batch')
    parser.add_argument('--atol', type=float, default=1e-4, help='Atol for ODE solver')
    parser.add_argument('--rtol', type=float, default=1e-4, help='Rtol for ODE solver')
    parser.add_argument('--type', type=str, default="cv-condition", help='training type')
    parser.add_argument('--tags', nargs='*', help='Tags for Wandb')
    
    return parser.parse_args()

args = parse_args()

n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
atom_types = np.arange(22)
atom_types[[0, 2, 3]] = 2
atom_types[1] = 0
atom_types[[11, 12, 13]] = 12
atom_types[[19, 20, 21]] = 20
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


# Set logging
wandb.init(
    project="tbg",
    entity="eddy26",
    config=vars(args),
    tags=["condition", "ECNF++"] + args.tags,
)
load_dir = f"./res/{args.date}/model"
checkpoint = torch.load(load_dir+"/tbg-final.pt")
save_dir = f"./res/{args.date}/result"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Set up
print(">> Setting up")
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
tbgcv_checkpoint = torch.load(load_dir+"/mlcv-final.pt")
tbgcv.load_state_dict(tbgcv_checkpoint)
tbgcv.eval()
cv_dimension = encoder_layers[-1]


if args.type == "label":
    cv_dimension = args.cv_dimension

prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)
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
    cv_dimension=cv_dimension,
    mode="egnn_dynamics",
    agg="sum",
)
bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator
)
flow = DiffEqFlow(dynamics=bb_dynamics)
flow.load_state_dict(checkpoint["model_state_dict"])
if args.type in ["repro"]:
    bg = BoltzmannGenerator(prior, flow, prior).cuda()
else:
    bg = BoltzmannGeneratorCV(prior, flow, prior).cuda()


print(">> Loading force estimator")
brute_force_estimator_fast = BruteForceEstimatorFast()
bb_dynamics._divergence_estimator = brute_force_estimator_fast
bg.flow._integrator_atol = args.atol
bg.flow._integrator_rtol = args.rtol
flow._use_checkpoints = False
flow._kwargs = {}

    

# Sampling ready
print(f">> Sampling")
n_samples = args.n_samples
n_sample_batches = args.n_sample_batches
latent_torch = torch.empty((n_samples * n_sample_batches, dim), dtype=torch.float32, device="cuda")
samples_torch = torch.empty((n_samples * n_sample_batches, dim), dtype=torch.float32, device="cuda")
dlogp_torch = torch.empty((n_samples * n_sample_batches, 1), dtype=torch.float32, device="cuda")
total_start_time = time.time()


# Conditioning
if args.type in ["cv-condition", "cfg"] and args.state in ["c5", "c7ax"]:
    state_path = f"../../simulation/data/alanine/{args.state}-tbg.pt"
    state_xyz = torch.load(state_path)['xyz']
    state_xyz = (state_xyz - 1.5508) / 0.6695
    state_heavy_atom_distance = coordinate2distance(state_xyz).cuda()
    state_heavy_atom_distance = state_heavy_atom_distance.repeat(n_samples, 1)
elif args.type in ["cv-condition-xyz", "cv-condition-xyz-ac"]:
    state_path = f"../../simulation/data/alanine/{args.state}.pt"
    state_xyz = torch.load(state_path)['xyz'].cuda()
    if args.state == "c7ax":
        reference_state_path = f"../../simulation/data/alanine/c5.pt"
        reference_state_xyz = torch.load(reference_state_path)['xyz'].cuda()
        state_xyz = kabsch(state_xyz, reference_state_xyz)[:, ALANINE_HEAVY_ATOM_IDX].reshape(1, -1)
    else:
        state_xyz = state_xyz[:, ALANINE_HEAVY_ATOM_IDX].reshape(1, -1)
    state_xyz = state_xyz.repeat(n_samples, 1)
elif args.type == "cv-condition-xyzhad":
    state_path = f"../../simulation/data/alanine/{args.state}.pt"
    state_xyz = torch.load(state_path)['xyz'][:, ALANINE_HEAVY_ATOM_IDX].reshape(1, -1).cuda()
    state_had = coordinate2distance(state_xyz).cuda()
    state_xyzhad = torch.cat([state_xyz, state_had], dim=1)
    state_xyzhad = state_xyzhad.repeat(n_samples, 1)
elif args.type == "label":
    if args.state == "c5":
        cv_condition = torch.ones((n_samples, cv_dimension)).cuda()
    elif args.state == "c7ax":
        cv_condition = torch.zeros((n_samples, cv_dimension)).cuda()
    else:
        raise ValueError("Invalid state for label condition")
elif args.state == "none":
    state_heavy_atom_distance = None
else:
    raise ValueError(f"Invalid state {args.state}")

print(f">> Condition state: {args.state}")
print(f">> MLCV: {tbgcv(state_xyz)[0]}")


# Torch sampling
pbar = tqdm.tqdm(
    range(n_sample_batches),
    desc="Sampling from BG: xx.xx seconds"
)
for i in pbar:
    with torch.no_grad():
        batch_start_time = time.time()
        # print("Start sampling at {}")
        if args.type in ["repro"]:
            samples, latent, dlogp = bg.sample(n_samples, with_latent=True, with_dlogp=True)
        elif args.type in ["label"]:
            samples, latent, dlogp = bg.sample(n_samples, cv_condition=cv_condition, with_latent=True, with_dlogp=True)
        elif args.type == "cv-condition":
            cv_condition = tbgcv(state_heavy_atom_distance)
            samples, latent, dlogp = bg.sample(n_samples, cv_condition=cv_condition, with_latent=True, with_dlogp=True)
        elif args.type in ["cv-condition-xyz", "cv-condition-xyz-ac"]:
            cv_condition = tbgcv(state_xyz)
            samples, latent, dlogp = bg.sample(n_samples, cv_condition=cv_condition, with_latent=True, with_dlogp=True)
        elif args.type == "cv-condition-xyzhad":
            cv_condition = tbgcv(state_xyzhad)
            samples, latent, dlogp = bg.sample(n_samples, cv_condition=cv_condition, with_latent=True, with_dlogp=True)
        batch_end_time = time.time()
        pbar.set_description(f"Sampling from BG: {batch_end_time - batch_start_time:.2f} seconds")
        # print(f"Batch {i} sampling time: {batch_end_time - batch_start_time:.2f} seconds")
        latent_torch[i * n_samples : (i + 1) * n_samples, :] = latent
        samples_torch[i * n_samples : (i + 1) * n_samples, :] = samples
        dlogp_torch[i * n_samples : (i + 1) * n_samples, :] = dlogp
        
    # if i % 10 == 0:    
    #     torch.save(latent_torch, f"{save_dir}/latent-{args.state}.pt")
    #     torch.save(samples_torch, f"{save_dir}/samples-{args.state}.pt")
    #     torch.save(dlogp_torch, f"{save_dir}/dlogp-{args.state}.pt")

# Original numpy concat sampling
# print(f"Start sampling with {filename}")
# latent_np = np.empty(shape=(0))
# samples_np = np.empty(shape=(0))
# dlogp_np = np.empty(shape=(0))
# for i in tqdm.tqdm(range(n_sample_batches)):
#     with torch.no_grad():
#         samples, latent, dlogp = bg.sample(n_samples, cv_condition=state_heavy_atom_distance, with_latent=True, with_dlogp=True)
#         latent_np = np.append(latent_np, latent.detach().cpu().numpy())
#         samples_np = np.append(samples_np, samples.detach().cpu().numpy())
#         dlogp_np = np.append(dlogp_np, as_numpy(dlogp))

#     latent_np = latent_np.reshape(-1, dim)
#     samples_np = samples_np.reshape(-1, dim)



# Save final results
print(f"Saved data at {save_dir}")
torch.save(latent_torch, f"{save_dir}/latent-{args.state}.pt")
torch.save(samples_torch, f"{save_dir}/samples-{args.state}.pt")
torch.save(dlogp_torch, f"{save_dir}/dlogp-{args.state}.pt")

total_end_time = time.time()
print(f"Total sampling time: {total_end_time - total_start_time:.2f} seconds")
print(f"Average time per batch: {(total_end_time - total_start_time) / n_sample_batches:.2f} seconds")

