cd ../

CUDA_VISIBLE_DEVICES=$1 python sample-tbgcv.py \
    --type repro \
    --tags sampling small repro \
    --hidden_dim 64 \
    --filename_tbg tbg-repro-v1 \
    --n_samples 400 \
    --n_sample_batches 5