cd ../

CUDA_VISIBLE_DEVICES=$1 python train-org.py

CUDA_VISIBLE_DEVICES=$1 python sample-org.py

