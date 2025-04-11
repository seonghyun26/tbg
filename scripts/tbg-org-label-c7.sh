cd ../

export TZ=Asia/Seoul
# CURRENT_DATE=$(date '+%m%d_%H%M%S')
CURRENT_DATE=
echo $CURRENT_DATE

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --date $CURRENT_DATE \
    --type cv-condition \
    --cv_dimension 2 \
    --tags training data-normalization cv-condition

CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type label \
    --tags sampling data-normalization cv-condition \
    --cv_dimension 2 \
    --state c5 \
    --n_samples 200 \
    --n_sample_batches 20

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --date $CURRENT_DATE \
    --scaling 1.4936519791 \
    --state c5 \
    --topology c5-tbg.pdb