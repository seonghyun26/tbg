cd ../

CUDA_VISIBLE_DEVICES=$1 python sample-tbgcv.py \
    --type repro \
    --tags sampling small repro \
    --hidden_dim 64 \
    --filename tbg-fixed \
    --n_samples 10 \
    --n_sample_batches 10