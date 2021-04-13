#!/bin/bash
BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CUDA_VISIBLE_DEVICES=3 python $BASE_PATH/train_model.py \
  --reference_type 'intrinsic' \
  --num_entity 2 \
  --excluded_entity '18' \
  --excluded_ralation 'all4' \
  --model_type 'film' \
  --num_modules 2 \
  --torch_random_seed 1 \
  --image_size 128 \
  --emoji_size 24 \
  --total_epochs 20 \
  --train_based_on_checkpoint 0 \
  --test 0 \
  --test_mode 'comp' \
  --checkpoint_dirname '2021-03-28 11:15:06.621355+02:00_film_num_modules_2' \
  --checkpoint_filename 'final' \
