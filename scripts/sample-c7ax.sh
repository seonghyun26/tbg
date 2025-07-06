cd ../

export TZ=Asia/Seoul

CURRENT_DATE=0522_174800
echo $CURRENT_DATE
CV_DIMENSION=1
STATE=c7ax


CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type cv-condition-xyz-ac \
    --tags training lag10 xyz \
    --hidden_dim 64 \
    --cv_dimension $CV_DIMENSION \
    --state $STATE \
    --n_samples 200 \
    --n_sample_batches 200

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --tags evaluation lag10 best\
    --type cv-condition-xyz-ac \
    --date $CURRENT_DATE \
    --cv_dimension $CV_DIMENSION \
    --scaling 1 \
    --state $STATE