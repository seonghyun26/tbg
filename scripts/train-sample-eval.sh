cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
# CURRENT_DATE=0411_221239
echo $CURRENT_DATE


CUDA_VISIBLE_DEVICES=$1 python train.py \
    --date $CURRENT_DATE \
    --type cv-condition \
    --tags training data-normalization cv-condition \
    --hidden_dim 64 


CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type cv-condition \
    --tags sampling data-normalization cv-condition \
    --hidden_dim 64 \
    --cv_dimension 2 \
    --state c5 \
    --n_samples 100 \
    --n_sample_batches 20

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --date $CURRENT_DATE \
    --scaling 1.4936519791 \
    --state c5 \
    --topology file

