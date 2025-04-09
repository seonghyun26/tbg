cd ../

FILE_NAME="repro-v1"

# CUDA_VISIBLE_DEVICES=$1 python train-org.py \
#     --tags training repro \
#     --repro True \
#     --ckpt_name fixed4 \

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --type repro \
#     --tags training data-normalization \
#     --hidden_dim 64 \
#     --ckpt_name $FILE_NAME

CUDA_VISIBLE_DEVICES=$1 python sample-tbgcv.py \
    --type repro \
    --tags sampling data-normalization \
    --hidden_dim 64 \
    --state none \
    --filename_tbg tbg-$FILE_NAME \
    --n_samples 500 \
    --n_sample_batches 40

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --file_name $FILE_NAME \
    --state none \
    --topology c5-tbg
