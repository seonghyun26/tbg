cd ../

CURRENT_DATE=$(date '+%m%d_%H%M%S')
echo $CURRENT_DATE

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --date $CURRENT_DATE \
    --type label \
    --cv_dimension 22 \
    --tags training data-normalization cv-label

CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type label \
    --tags sampling small data-normalization cv-label \
    --cv_dimension 22 \
    --state c5 \
    --n_samples 100 \
    --n_sample_batches 40

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --date $CURRENT_DATE \
    --scaling 1.4936519791 \
    --state c5 \
    --topology c5-tbg.pdb
