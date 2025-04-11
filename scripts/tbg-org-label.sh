cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
# CURRENT_DATE=0411_024308
echo $CURRENT_DATE

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --date $CURRENT_DATE \
    --type label \
    --cv_dimension 1 \
    --tags training data-normalization org-label

CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type label \
    --tags sampling data-normalization org-label \
    --cv_dimension 1 \
    --state c5 \
    --n_samples 100 \
    --n_sample_batches 10

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --date $CURRENT_DATE \
    --scaling 10 \
    --state c5 \
    --topology c5-tbg.pdb
