cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
# CURRENT_DATE=0410_061408
echo $CURRENT_DATE

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --date $CURRENT_DATE \
    --current_xyz ../../simulation/dataset/alanine/300.0/tbg-10n-half/current-xyz.pt \
    --type repro \
    --tags training data-normalization half

CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type repro \
    --tags sampling data-normalization half \
    --state none \
    --n_samples 100 \
    --n_sample_batches 20

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --date $CURRENT_DATE \
    --scaling 1.4936519791 \
    --state none \
    --topology c5-tbg.pdb

    # --scaling 1.1238480557 \