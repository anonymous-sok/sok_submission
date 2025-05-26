#!/bin/bash
BASE_PATH="../../src/CVPR22_NICGSlowDown"

# 1
python train_detector.py \
    --input_file ${BASE_PATH}/adv_train/0_L2_BEST_coco_mobilenet_rnn.adv \
    --test_file ${BASE_PATH}/adv/0_L2_BEST_coco_mobilenet_rnn.adv \
    --test_jpeg_file ${BASE_PATH}/adv/jpeg/0_L2_BEST_coco_mobilenet_rnn.adv \
    --test_spatial_file ${BASE_PATH}/adv/spatial/0_L2_BEST_coco_mobilenet_rnn.adv \
    --batch_size 64

# 2
python train_detector.py \
    --input_file ${BASE_PATH}/adv_train/0_L2_BEST_coco_resnet_lstm.adv \
    --test_file ${BASE_PATH}/adv/0_L2_BEST_coco_resnet_lstm.adv \
    --test_jpeg_file ${BASE_PATH}/adv/jpeg/0_L2_BEST_coco_resnet_lstm.adv \
    --test_spatial_file ${BASE_PATH}/adv/spatial/0_L2_BEST_coco_resnet_lstm.adv \
    --batch_size 64

# 3
python train_detector.py \
    --input_file ${BASE_PATH}/adv_train/0_L2_BEST_flickr8k_googlenet_rnn.adv \
    --test_file ${BASE_PATH}/adv/0_L2_BEST_flickr8k_googlenet_rnn.adv \
    --test_jpeg_file ${BASE_PATH}/adv/jpeg/0_L2_BEST_flickr8k_googlenet_rnn.adv \
    --test_spatial_file ${BASE_PATH}/adv/spatial/0_L2_BEST_flickr8k_googlenet_rnn.adv \
    --batch_size 64

# 4
python train_detector.py \
    --input_file ${BASE_PATH}/adv_train/0_L2_BEST_flickr8k_resnext_lstm.adv \
    --test_file ${BASE_PATH}/adv/0_L2_BEST_flickr8k_resnext_lstm.adv \
    --test_jpeg_file ${BASE_PATH}/adv/jpeg/0_L2_BEST_flickr8k_resnext_lstm.adv \
    --test_spatial_file ${BASE_PATH}/adv/spatial/0_L2_BEST_flickr8k_resnext_lstm.adv \
    --batch_size 64

# 5
python train_detector.py \
    --input_file ${BASE_PATH}/adv_train/0_Linf_BEST_coco_mobilenet_rnn.adv \
    --test_file ${BASE_PATH}/adv/0_Linf_BEST_coco_mobilenet_rnn.adv \
    --test_jpeg_file ${BASE_PATH}/adv/jpeg/0_Linf_BEST_coco_mobilenet_rnn.adv \
    --test_spatial_file ${BASE_PATH}/adv/spatial/0_Linf_BEST_coco_mobilenet_rnn.adv \
    --batch_size 64

# 6
python train_detector.py \
    --input_file ${BASE_PATH}/adv_train/0_Linf_BEST_coco_resnet_lstm.adv \
    --test_file ${BASE_PATH}/adv/0_Linf_BEST_coco_resnet_lstm.adv \
    --test_jpeg_file ${BASE_PATH}/adv/jpeg/0_Linf_BEST_coco_resnet_lstm.adv \
    --test_spatial_file ${BASE_PATH}/adv/spatial/0_Linf_BEST_coco_resnet_lstm.adv \
    --batch_size 64

# 7
python train_detector.py \
    --input_file ${BASE_PATH}/adv_train/0_Linf_BEST_flickr8k_googlenet_rnn.adv \
    --test_file ${BASE_PATH}/adv/0_Linf_BEST_flickr8k_googlenet_rnn.adv \
    --test_jpeg_file ${BASE_PATH}/adv/jpeg/0_Linf_BEST_flickr8k_googlenet_rnn.adv \
    --test_spatial_file ${BASE_PATH}/adv/spatial/0_Linf_BEST_flickr8k_googlenet_rnn.adv \
    --batch_size 64

# 8
python train_detector.py \
    --input_file ${BASE_PATH}/adv_train/0_Linf_BEST_flickr8k_resnext_lstm.adv \
    --test_file ${BASE_PATH}/adv/0_Linf_BEST_flickr8k_resnext_lstm.adv \
    --test_jpeg_file ${BASE_PATH}/adv/jpeg/0_Linf_BEST_flickr8k_resnext_lstm.adv \
    --test_spatial_file ${BASE_PATH}/adv/spatial/0_Linf_BEST_flickr8k_resnext_lstm.adv \
    --batch_size 64