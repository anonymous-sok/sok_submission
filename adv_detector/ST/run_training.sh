#!/bin/bash
BASE_PATH="../../src/SlowTrack"
BASE_PATH_DEFENSE="../../src/mitigation"


# Benign Examples
python train_detector.py \
    --clean_folder ${BASE_PATH}/dataset \
    --adv_folder ${BASE_PATH}/botsort_stra_0.25 \
    --inference_jpeg_folder ${BASE_PATH_DEFENSE}/img_clean_jpeg/SlowTrack \
    --inference_jpeg_label 0 \
    --inference_spatial_folder ${BASE_PATH_DEFENSE}/img_clean_spatial/SlowTrack \
    --inference_spatial_label 0 \
    --batch_size 64 \
    --max_pairs 400 \
    --train_ratio 0.5

python train_detector.py \
    --clean_folder ${BASE_PATH}/dataset \
    --adv_folder ${BASE_PATH}/botsort_stra_0.5 \
    --inference_jpeg_folder ${BASE_PATH_DEFENSE}/img_clean_jpeg/SlowTrack \
    --inference_jpeg_label 0 \
    --inference_spatial_folder ${BASE_PATH_DEFENSE}/img_clean_spatial/SlowTrack\
    --inference_spatial_label 0 \
    --batch_size 64 \
    --max_pairs 400 \
    --train_ratio 0.5

python train_detector.py \
    --clean_folder ${BASE_PATH}/dataset \
    --adv_folder ${BASE_PATH}/botsort_stra_0.75 \
    --inference_jpeg_folder ${BASE_PATH_DEFENSE}/img_clean_jpeg/SlowTrack\
    --inference_jpeg_label 0 \
    --inference_spatial_folder ${BASE_PATH_DEFENSE}/img_clean_spatial/SlowTrack \
    --inference_spatial_label 0 \
    --batch_size 64 \
    --max_pairs 400 \
    --train_ratio 0.5


# Adv Examples
python train_detector.py \
    --clean_folder ${BASE_PATH}/dataset \
    --adv_folder ${BASE_PATH}/botsort_stra_0.25 \
    --inference_jpeg_folder ${BASE_PATH}/botsort_stra_0.25/jpeg \
    --inference_jpeg_label 1 \
    --inference_spatial_folder ${BASE_PATH}/botsort_stra_0.25/spatial \
    --inference_spatial_label 1 \
    --batch_size 64 \
    --max_pairs 400 \
    --train_ratio 0.5

python train_detector.py \
    --clean_folder ${BASE_PATH}/dataset \
    --adv_folder ${BASE_PATH}/botsort_stra_0.5 \
    --inference_jpeg_folder ${BASE_PATH}/botsort_stra_0.5/jpeg \
    --inference_jpeg_label 1 \
    --inference_spatial_folder ${BASE_PATH}/botsort_stra_0.5/spatial \
    --inference_spatial_label 1 \
    --batch_size 64 \
    --max_pairs 400 \
    --train_ratio 0.5

python train_detector.py \
    --clean_folder ${BASE_PATH}/dataset \
    --adv_folder ${BASE_PATH}/botsort_stra_0.75 \
    --inference_jpeg_folder ${BASE_PATH}/botsort_stra_0.75/jpeg\
    --inference_jpeg_label 1 \
    --inference_spatial_folder ${BASE_PATH}/botsort_stra_0.75/spatial \
    --inference_spatial_label 1 \
    --batch_size 64 \
    --max_pairs 400 \
    --train_ratio 0.5