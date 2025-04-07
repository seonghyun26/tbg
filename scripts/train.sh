cd ../


CUDA_VISIBLE_DEVICES=$1 python train.py \
    --tags training \
    --hidden_dim 64 \
    --ckpt_name fixed6 \
    --data_xyz_path ../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg2.pt \
    --data_distance_path ../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt 

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --tags larger-data \
#     --data_type 10n-large \
#     --data_xyz_path ../../simulation/dataset/alanine/300.0/timelag-10n-lag10-large/xyz-tbg.pt \
#     --data_distance_path ../../simulation/dataset/alanine/300.0/timelag-10n-lag10-large/distance-tbg.pt

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --data_type 1n \
#     --data_xyz_path ../../simulation/dataset/alanine/300.0/timelag-1n-v1/xyz-tbg.pt \
#     --data_distance_path ../../simulation/dataset/alanine/300.0/timelag-1n-v1/distance-tbg.pt

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --tags optimizer-test \
#     --data_type 10n \
#     --data_xyz_path ../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg.pt \
#     --data_distance_path ../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt

# CUDA_VISIBLE_DEVICES=$1 python train-cl.py \
#     --lambda_state_diff 0.1 \
#     --tags state-mlcv-cl \
#     --data_type 10n \
#     --data_xyz_path ../../simulation/dataset/alanine/300.0/timelag-10n-v1/xyz-tbg.pt \
#     --data_distance_path ../../simulation/dataset/alanine/300.0/timelag-10n-v1/distance-tbg.pt

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --data_type 250n \
#     --data_xyz_path ../../simulation/dataset/alanine/300.0/timelag-250n-v1/xyz-tbg.pt \
#     --data_distance_path ../../simulation/dataset/alanine/300.0/timelag-250n-v1/distance-tbg.pt