cd ../

CUDA_VISIBLE_DEVICES=$1 python sample-tbgcv.py \
    --tags sampling small \
    --hidden_dim 256 \
    --state c7ax \
    --filename tbg-fixed \
    --n_samples 10 \
    --n_sample_batches 10