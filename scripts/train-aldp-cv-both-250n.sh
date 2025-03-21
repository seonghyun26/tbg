cd ../


CUDA_VISIBLE_DEVICES=$1 python train-both.py \
    --data_type 250n \
    --data_xyz_path ../../simulation/dataset/alanine/300.0/timelag-250n-v1/xyz-tbg.pt \
    --data_distance_path ../../simulation/dataset/alanine/300.0/timelag-250n-v1/distance-tbg.pt
