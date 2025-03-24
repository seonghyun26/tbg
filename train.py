import os
import torch
import wandb
import numpy as np
import argparse

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, BoltzmannGeneratorCV, MeanFreeNormalDistribution
from tbg.modelwithcv import EGNN_AD2_CV
from bgflow import BlackBoxDynamics, BruteForceEstimator

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train TBG model')
    parser.add_argument('--data_xyz_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg.pt", help='Path to xyz data file')
    parser.add_argument('--data_distance_path', type=str, default= "../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt", help='Path to distance data file')
    parser.add_argument('--data_type', type=str, default= "1n", help='data type')
    parser.add_argument('--sample_epoch', type=int, default= "500", help='epoch interval for sampling')
    return parser.parse_args()

args = parse_args()

n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions
n_samples = 100
n_sample_batches = 400

wandb.init(
    project="tbg",
    entity="eddy26",
    config={
        "data": args.data_xyz_path,
    },
    tags=["condition"]
)

# atom types for backbone
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
flow._dynamics._dynamics._dynamics_function.cv.condition = True
bg = BoltzmannGeneratorCV(prior, flow, prior).cuda()


n_batch = 256
data_xyz_path = args.data_xyz_path
data_xyz = torch.load(data_xyz_path)
data_distance_path = args.data_distance_path
data_distance = torch.load(data_distance_path)
batch_iter = IndexBatchIterator(len(data_xyz), n_batch)
# sample_batch_iter = IndexBatchIterator(len(data_xyz), n_batch // 2)
# Dataset size: data num * (current, time lag, distance, time lag distance) * coordinates (66)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)
n_epochs = 1000
PATH_last = f"models/tbgcv/{args.data_type}"
if not os.path.exists(PATH_last):
    os.makedirs(PATH_last)
PATH_last = f"models/tbgcv/{args.data_type}/"


sigma = 0.01
loss = 0
pbar = tqdm(range(n_epochs), desc = f"Loss: {loss:.4f}",)
for epoch in pbar:
    if epoch == 500:
        for g in optim.param_groups:
            g["lr"] = 5e-5

    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        # x1 = data_xyz[idx][:, 0].cuda()
        x1 = data_xyz[idx][:, 1].cuda()
        x1_distance = data_distance[idx][:, 0].cuda()
        # x1_timelag_distance = data_distance[idx][:, 1].cuda()
        batchsize = x1.shape[0]
        t = torch.rand(batchsize, 1).cuda()
        x0 = prior_cpu.sample(batchsize).cuda()

        # calculate regression loss
        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        noise = prior.sample(batchsize)
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        
        # Flow
        # vt = flow._dynamics._dynamics._dynamics_function(t, x)
        vt = flow._dynamics._dynamics._dynamics_function(t, x, x1_distance)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optim.step()
    pbar.set_description(f"Loss: {loss:.4f}")
    
    wandb.log({
        "loss": loss.item(),
    }, step=epoch)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}")
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last + f"/_tbg_{epoch}.pt",
        )
        torch.save(
            flow._dynamics._dynamics._dynamics_function.cv.state_dict(),
            PATH_last + f"/_mlcv_{epoch}.pt",   
        )
        
    # if epoch % args.sample_epoch == 0 and epoch != 0:
    #     print(f"Sampling test at epoch {epoch}")
    #     latent_np = np.empty(shape=(0))
    #     samples_np = np.empty(shape=(0))
    #     dlogp_np = np.empty(shape=(0))
        
    #     for it, idx in enumerate(tqdm(
    #         sample_batch_iter,
    #         desc = "Sampling from BG"
    #     )):
    #         # Load data
    #         x1 = data_xyz[idx][:, 1].cuda()
    #         x1_distance = data_distance[idx][:, 0].cuda()
    #         batchsize = x1.shape[0]
    #         t = torch.rand(batchsize, 1).cuda()
    #         x0 = prior_cpu.sample(batchsize).cuda()

    #         # Check reconstruction
    #         with torch.no_grad():
    #             samples, latent, dlogp = bg.sample(n_samples=batchsize, prior_samples=x0, cv_condition=x1_distance, with_latent=True, with_dlogp=True)
    #             latent_np = np.append(latent_np, latent.detach().cpu().numpy())
    #             samples_np = np.append(samples_np, samples.detach().cpu().numpy())
    #             dlogp_np = np.append(dlogp_np, dlogp.cpu().numpy())

    #         latent_np = latent_np.reshape(-1, dim)
    #         samples_np = samples_np.reshape(-1, dim)
    #         np.savez(
    #             f"result_data/tbgcv/sample_{epoch}",
    #             latent_np=latent_np,
    #             samples_np=samples_np,
    #             dlogp_np=dlogp_np,
    #             x1=x1.cpu().numpy(),
    #     )

print(f">> Final epoch {epoch}")
torch.save(
    {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
    },
    PATH_last + f"/_tbg_{epoch}.pt",
)
torch.save(
    flow._dynamics._dynamics._dynamics_function.cv.state_dict(),
    PATH_last + f"/_mlcv_{epoch}.pt",   
)

latent_np = np.empty(shape=(0))
samples_np = np.empty(shape=(0))
dlogp_np = np.empty(shape=(0))

# Check reconstruction loss
bg = BoltzmannGeneratorCV(prior, flow, prior).cuda()
n_samples = batchsize
n_sample_batches = 256
latent_np = np.empty(shape=(0))
samples_np = np.empty(shape=(0))
dlogp_np = np.empty(shape=(0))
with torch.no_grad():
    # x0_with_distance = torch.cat([x0, x1_distance], dim=1)
    samples, latent, dlogp = bg.sample(n_samples, prior_samples=x0, cv_condition=x1_distance, with_latent=True, with_dlogp=True)
    latent_np = np.append(latent_np, latent.detach().cpu().numpy())
    samples_np = np.append(samples_np, samples.detach().cpu().numpy())
    dlogp_np = np.append(dlogp_np, dlogp.cpu().numpy())

latent_np = latent_np.reshape(-1, dim)
samples_np = samples_np.reshape(-1, dim)
np.savez(
    f"result_data/tbgcv/sample_{epoch}",
    latent_np=latent_np,
    samples_np=samples_np,
    dlogp_np=dlogp_np,
    x1=x1.cpu().numpy(),
)


wandb.finish()