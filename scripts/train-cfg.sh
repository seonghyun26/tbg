cd ../

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --cv_condition_scale 1.0 \
    --tags training data-normalization cv-not-normalized cfg \
    --hidden_dim 64 \
    --ckpt_name fixed167 \
    --cfg True