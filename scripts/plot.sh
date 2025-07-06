CKPT=0519_171522

cd ../
CUDA_VISIBLE_DEVICES=$1 python plot-ram.py \
    --type cv-condition-xyz-ac \
    --date $CKPT \
    --cv_dimension 1