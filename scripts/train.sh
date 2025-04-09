cd ../


CUDA_VISIBLE_DEVICES=$1 python train.py \
    --type repro \
    --tags training data-normalization\
    --hidden_dim 64 \
    --ckpt_name repro-v1
