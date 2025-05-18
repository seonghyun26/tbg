cd ../

export TZ=Asia/Seoul

CURRENT_DATE=0505_232520
echo $CURRENT_DATE
CV_DIMENSION=1


# CUDA_VISIBLE_DEVICES=$1 python sample.py \
#     --date $CURRENT_DATE \
#     --type cv-condition-xyz \
#     --tags training lag0 xyz \
#     --hidden_dim 64 \
#     --cv_dimension $CV_DIMENSION \
#     --state c5 \
#     --n_samples 200 \
#     --n_sample_batches 200

# CUDA_VISIBLE_DEVICES=$1 python eval.py \
#     --type cv-condition-xyz \
#     --date $CURRENT_DATE \
#     --cv_dimension $CV_DIMENSION \
#     --scaling 1 \
#     --state c5 \
#     --topology file


CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type cv-condition-xyzhad \
    --tags training lag10 xyz \
    --hidden_dim 64 \
    --cv_dimension $CV_DIMENSION \
    --state c7ax \
    --n_samples 200 \
    --n_sample_batches 200

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --date $CURRENT_DATE \
    --type cv-condition-xyzhad \
    --cv_dimension $CV_DIMENSION \
    --scaling 1 \
    --state c7ax \
    --topology file