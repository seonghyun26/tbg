cd ../

export TZ=Asia/Seoul
CURRENT_DATE=abl-ac
echo $CURRENT_DATE
CV_DIMENSION=1

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --tags training lag10 \
    --current_xyz ../../simulation/dataset/alanine/300.0/10nano-v2/current-xyz-aligned.pt \
    --current_distance ../../simulation/dataset/alanine/300.0/10nano-v2/current-distance.pt \
    --timelag_xyz ../../simulation/dataset/alanine/300.0/10nano-v2/timelag-xyz.pt \
    --ac_timelag_xyz ../../simulation/dataset/alanine/300.0/10nano-v2/timelag-xyz-aligned.pt \
    --date $CURRENT_DATE \
    --ac_loss_lambda 0.000 \
    --n_epochs 1000 \
    --dropout 0.5 \
    --type cv-condition-xyz-ac \
    --cv_dimension $CV_DIMENSION \
    --hidden_dim 64