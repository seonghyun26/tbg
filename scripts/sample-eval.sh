cd ../

FILE_NAME="tbg-fixed5"
state=c7ax

CUDA_VISIBLE_DEVICES=$1 python sample-tbgcv.py \
    --tags sampling \
    --tags small \
    --state $state \
    --filename $FILE_NAME \
    --n_samples 1 \
    --n_sample_batches 1


CUDA_VISIBLE_DEVICES=$1 python eval.py \
  --file_name $FILE_NAME \
  --state $state