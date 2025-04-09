cd ../

FILE_NAME="tbg-label2"


CUDA_VISIBLE_DEVICES=$1 python train.py \
    --tags training data-normalization cv-label \
    --hidden_dim 64 \
    --ckpt_name $FILE_NAME

CUDA_VISIBLE_DEVICES=$1 python sample-tbgcv.py \
    --type label \
    --tags sampling small data-normalization cv-label \
    --hidden_dim 64 \
    --state c5 \
    --filename $FILE_NAME \
    --n_samples 100 \
    --n_sample_batches 100

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --file_name $FILE_NAME \
    --state c5 \
    --topology c5-tbg
