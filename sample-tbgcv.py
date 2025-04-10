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
from tbg.modelwithcv import EGNN_AD2_CV, TBGCV
from tbg.models2 import EGNN_dynamics_AD2_cat, EGNN_dynamics_AD2_cat_CV

from src.force import BruteForceEstimatorFast

ALANINE_HEAVY_ATOM_IDX_TBG = [0, 4, 5, 6, 8, 10, 14, 15, 16, 18]



def coordinate2distance(
    positions
): 
    distance_list = []
    
    for position in positions:
        position = position.reshape(-1, 3)
        heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX_TBG]
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
    parser.add_argument('--filename_tbg', type=str, default= "tbgcv", help='checkpoint name for tbg')
    parser.add_argument('--filename_mlcv', type=str, default= "mlcv", help='checkpoint name for mlcv')
    parser.add_argument('--hidden_dim', type=int, default="256", help='hidden dimension of EGNN')
    parser.add_argument('--cv_dimension', type=int, default="2", help='cv dimension')
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
scaling = 10
atom_types = np.arange(22)
atom_types[[1, 2, 3]] = 2
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
filename = args.filename_tbg
PATH_last = f"models/{filename}.pt"
checkpoint = torch.load(PATH_last)
if not os.path.exists(f"result_data/{filename}/"):
    os.makedirs(f"result_data/{filename}/")


# Set up
print(">> Setting up")
if args.type == "cv-condition":
    encoder_layers = [45, 30, 30, args.cv_dimension]
    cv_dimension = encoder_layers[-1]
    tbgcv = TBGCV(encoder_layers=encoder_layers).cuda()
    tbgcv.eval()
    tbgcv_ckpt = torch.load(f"models/{args.filename_mlcv}.pt")
    tbgcv.load_state_dict(tbgcv_ckpt)
elif args.type == "label":
    cv_dimension = args.cv_dimension
else:
    cv_dimension = 0

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
print(f">> Sampling with {filename} for {args.state}")
n_samples = args.n_samples
n_sample_batches = args.n_sample_batches
# latent_np = np.empty(shape=(0))
# samples_np = np.empty(shape=(0))
# dlogp_np = np.empty(shape=(0))
latent_torch = torch.empty((n_samples * n_sample_batches, dim), dtype=torch.float32).cuda()
samples_torch = torch.empty((n_samples * n_sample_batches, dim), dtype=torch.float32).cuda()
dlogp_torch = torch.empty((n_samples * n_sample_batches, 1), dtype=torch.float32).cuda()
total_start_time = time.time()


# Conditioning
if args.state in ["c5", "c7ax"]:
    state_path = f"../../simulation/data/alanine/{args.state}-tbg.pt"
    state_xyz = torch.load(state_path)['xyz']
    state_xyz = (state_xyz - 1.5508) / 0.6695
    state_heavy_atom_distance = coordinate2distance(state_xyz).cuda()
    state_heavy_atom_distance = state_heavy_atom_distance.repeat(n_samples, 1)
elif args.state == "none":
    state_heavy_atom_distance = None
else:
    data_xyz_path = args.data_xyz_path
    data_xyz = torch.load(data_xyz_path)
    data_distance_path = args.data_distance_path
    data_distance = torch.load(data_distance_path)
    batch_iter = IndexBatchIterator(len(data_xyz), n_samples)
    rmsd = []
    heavy_atom_distance_avg = []
    heavy_atom_distance_difference = []
    mse = torch.nn.MSELoss()



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
            cv_condition = torch.ones((n_samples, cv_dimension)).cuda()
            samples, latent, dlogp = bg.sample(n_samples, cv_condition=cv_condition, with_latent=True, with_dlogp=True)
        else:
            cv_condition = tbgcv(state_heavy_atom_distance)
            samples, latent, dlogp = bg.sample(n_samples, cv_condition=cv_condition, with_latent=True, with_dlogp=True)
        batch_end_time = time.time()
        pbar.set_description(f"Sampling from BG: {batch_end_time - batch_start_time:.2f} seconds")
        # print(f"Batch {i} sampling time: {batch_end_time - batch_start_time:.2f} seconds")
        latent_torch[i * n_samples : (i + 1) * n_samples, :] = latent
        samples_torch[i * n_samples : (i + 1) * n_samples, :] = samples
        dlogp_torch[i * n_samples : (i + 1) * n_samples, :] = dlogp
        
        
        torch.save(latent_torch, f"result_data/{filename}/latent-{args.state}.pt")
        torch.save(samples_torch, f"result_data/{filename}/samples-{args.state}.pt")
        torch.save(dlogp_torch, f"result_data/{filename}/dlogp-{args.state}.pt")

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



# TODO: Sampling with guidance scale in case of CFG
# for it, idx in enumerate(tqdm.tqdm(batch_iter, desc="Sampling from BG")):
#     x1 = data_xyz[idx][:, 1].cuda()
#     x1_distance = data_distance[idx][:, 0].cuda()
#     heavy_atom_distance_avg.append(x1_distance.mean())
#     n_samples = x1.shape[0]
#     with torch.no_grad():
#         samples, latent, dlogp = bg.sample(n_samples, cv_condition=x1_distance, with_latent=True, with_dlogp=True)
#         latent_np = np.append(latent_np, latent.detach().cpu().numpy())
#         samples_np = np.append(samples_np, samples.detach().cpu().numpy())
#         dlogp_np = np.append(dlogp_np, as_numpy(dlogp))
    
#     rmsd_list = []
#     for i in range(n_samples):
#         sample = samples[i].reshape(-1, 3)
#         reference = x1[i].reshape(-1, 3)
#         rmsd_list.append(kabsch_rmsd(reference, sample))
    
#     rmsd.append(torch.stack(rmsd_list).mean())
#     heavy_atom_distance_difference.append(mse(coordinate2distance(samples), x1_distance))
# wandb.log({
#     "rmsd": torch.stack(rmsd).mean(),
#     "heavy_atom_distance_difference": torch.stack(heavy_atom_distance_difference).mean(),
#     "heavy_atom_distance_avg": torch.stack(heavy_atom_distance_avg).mean(),
#     "heavy_atom_distance_std": torch.stack(heavy_atom_distance_avg).std(),
# })
# print(f"RMSD: {torch.stack(rmsd).mean()}")
# print(f"Heavy atom distance difference: {torch.stack(heavy_atom_distance_difference).mean()}")
# print(f"Heavy atom distance avg: {torch.stack(heavy_atom_distance_avg).mean()}")
# print(f"Heavy atom distance std: {torch.stack(heavy_atom_distance_avg).std()}")


# Save final results
print(f"Saved data at result_data/{filename}/")
torch.save(latent_torch, f"result_data/{filename}/latent-{args.state}.pt")
torch.save(samples_torch, f"result_data/{filename}/samples-{args.state}.pt")
torch.save(dlogp_torch, f"result_data/{filename}/dlogp-{args.state}.pt")

total_end_time = time.time()
print(f"Total sampling time: {total_end_time - total_start_time:.2f} seconds")
print(f"Average time per batch: {(total_end_time - total_start_time) / n_sample_batches:.2f} seconds")

