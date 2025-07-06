cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
# CURRENT_DATE=0519_104717
echo $CURRENT_DATE
CV_DIMENSION=2

CUDA_VISIBLE_DEVICES=$1 python train-control.py \
    --tags training lag10 \
    --current_xyz ../../simulation/dataset/alanine/300.0/10nano-v2/current-xyz-aligned.pt \
    --current_distance ../../simulation/dataset/alanine/300.0/10nano-v2/current-distance.pt \
    --timelag_xyz ../../simulation/dataset/alanine/300.0/10nano-v2/timelag-xyz.pt \
    --ac_timelag_xyz ../../simulation/dataset/alanine/300.0/10nano-v2/timelag-xyz-aligned.pt \
    --date $CURRENT_DATE \
    --n_epochs 1000 \
    --type cv-condition-xyz \
    --cv_dimension $CV_DIMENSION \
    --hidden_dim 64

# CUDA_VISIBLE_DEVICES=$1 python plot-ram.py \
#     --type cv-condition-xyz-ac \
#     --date $CURRENT_DATE \
#     --cv_dimension $CV_DIMENSION 

# CUDA_VISIBLE_DEVICES=$1 python sample.py \
#     --date $CURRENT_DATE \
#     --type cv-condition-xyz-ac \
#     --tags lag10 xyz kabsch \
#     --hidden_dim 64 \
#     --cv_dimension $CV_DIMENSION \
#     --state c5 \
#     --n_samples 10 \
#     --n_sample_batches 10

# CUDA_VISIBLE_DEVICES=$1 python eval.py \
#     --type cv-condition-xyz-ac \
#     --date $CURRENT_DATE \
#     --cv_dimension $CV_DIMENSION \
#     --scaling 1 \
#     --state c5 \
#     --topology file


# CUDA_VISIBLE_DEVICES=$1 python sample.py \
#     --date $CURRENT_DATE \
#     --type cv-condition-xyz \
#     --tags training lag0 xyz \
#     --hidden_dim 64 \
#     --cv_dimension $CV_DIMENSION \
#     --state c7ax \
#     --n_samples 100 \
#     --n_sample_batches 20

# CUDA_VISIBLE_DEVICES=$1 python eval.py \
#     --date $CURRENT_DATE \
#     --cv_dimension $CV_DIMENSION \
#     --scaling 1 \
#     --state c7ax \
#     --topology file