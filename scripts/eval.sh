cd ../

CUDA_VISIBLE_DEVICES=$1 python eval.py \
  --file_name tbg-repro-v1 \
  --state none