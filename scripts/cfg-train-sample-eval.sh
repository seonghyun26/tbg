cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
CURRENT_DATE=0415_004136
echo $CURRENT_DATE
CV_DIMENSION=1
TYPE=cfg


# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --date $CURRENT_DATE \
#     --type $TYPE \
#     --tags training data-normalization $TYPE \
#     --cv_dimension $CV_DIMENSION \
#     --hidden_dim 64 


CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type $TYPE \
    --tags sampling data-normalization $TYPE \
    --hidden_dim 64 \
    --cv_dimension $CV_DIMENSION \
    --state c5 \
    --n_samples 100 \
    --n_sample_batches 20

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --tags sampling data-normalization $TYPE \
    --date $CURRENT_DATE \
    --cv_dimension $CV_DIMENSION \
    --scaling 1.4936519791 \
    --state c5 \
    --topology file

