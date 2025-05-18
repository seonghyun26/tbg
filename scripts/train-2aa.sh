cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
echo $CURRENT_DATE


CUDA_VISIBLE_DEVICES=$1 python train-2aa.py \
    --date $CURRENT_DATE \
    --tags reproduce