cd ../

# FILE_NAME="tbg-integrated"
FILE_NAME="tbg-fixed167"
state=c5
# state=c7ax

CUDA_VISIBLE_DEVICES=$1 python sample-tbgcv.py \
    --tags sampling small data-normalization\
    --hidden_dim 64 \
    --state $state \
    --filename $FILE_NAME \
    --n_samples 40 \
    --n_sample_batches 10 


CUDA_VISIBLE_DEVICES=$1 python eval.py \
  --file_name $FILE_NAME \
  --state $state \
  --topology c5-tbg