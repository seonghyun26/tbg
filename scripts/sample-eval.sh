cd ../

CURRENT_DATE=0414_151903
STATE=c7ax
echo $CURRENT_DATE
echo $STATE

CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --date $CURRENT_DATE \
    --type cv-condition \
    --tags sampling data-normalization cv-condition \
    --hidden_dim 64 \
    --cv_dimension 2 \
    --state $STATE \
    --n_samples 100 \
    --n_sample_batches 20

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    --date $CURRENT_DATE \
    --scaling 1.4936519791 \
    --state $STATE \
    --topology file

