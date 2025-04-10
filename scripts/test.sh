# !/bin/bash

export TZ=Asia/Seoul

current_date=$(date '+%m%d_%H%M%S')

echo $current_date

python test.py \
  --date $current_date