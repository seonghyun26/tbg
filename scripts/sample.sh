cd ../

CUDA_VISIBLE_DEVICES=$1 python sample-tbgcv.py \
    --tags sampling \
    --tags small \
    --state c7ax \
    --filename tbg-fixed3