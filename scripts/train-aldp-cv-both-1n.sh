cd ../


CUDA_VISIBLE_DEVICES=$1 python train-both.py \
    --data_type 1n \
    --data_xyz_path ../../simulation/dataset/alanine/300.0/timelag-1n-v1/xyz-tbg.pt \
    --data_distance_path ../../simulation/dataset/alanine/300.0/timelag-1n-v1/distance-tbg.pt
