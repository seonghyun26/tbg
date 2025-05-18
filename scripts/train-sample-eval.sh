cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
# CURRENT_DATE=0414_171040
echo $CURRENT_DATE
CV_DIMENSION=1


CUDA_VISIBLE_DEVICES=$1 python train.py \
    --timelag_xyz ../../simulation/dataset/alanine/300.0/tbg-10n-lag30/timelag-xyz.pt \
    --current_distance ../../simulation/dataset/alanine/300.0/tbg-10n-lag30/current-distance.pt \
    --date $CURRENT_DATE \
    --type cv-condition \
    --tags training data-normalization \
    --cv_dimension $CV_DIMENSION \
    --hidden_dim 64 


CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type cv-condition \
    --tags sampling data-normalization cv-condition \
    --hidden_dim 64 \
    --cv_dimension $CV_DIMENSION \
    --state c5 \
    --n_samples 100 \
    --n_sample_batches 20

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --date $CURRENT_DATE \
    --cv_dimension $CV_DIMENSION \
    --scaling 1.4936519791 \
    --state c5 \
    --topology file

