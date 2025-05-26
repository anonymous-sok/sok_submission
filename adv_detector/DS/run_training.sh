#!/bin/bash
BASE_PATH="../../src/DeepSloth/complete_datasets/cifar10/cifar10_resnet56_sdn_ic_only_1000samples"
BASE_PATH_DEFENSE="../../src/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only"

python train_detector.py \
  --clean_train ${BASE_PATH}/clean_train_1000.pickle \
  --adv_train ${BASE_PATH}/adv_persample_train_l2_1000.pickle \
  --clean_valid ${BASE_PATH}/clean_valid_1000.pickle \
  --adv_valid ${BASE_PATH}/adv_persample_valid_l2_1000.pickle \
  --clean_jpeg ${BASE_PATH_DEFENSE}/jpeg/ours_l2_clean.pickle \
  --adv_jpeg ${BASE_PATH_DEFENSE}/spatial/ours_l2_persample.pickle \
  --clean_spatial ${BASE_PATH_DEFENSE}/jpeg/ours_l2_clean.pickle \
  --adv_spatial ${BASE_PATH_DEFENSE}/spatial/ours_l2_persample.pickle \
  --model resnet50

python train_detector.py \
  --clean_train ${BASE_PATH}/clean_train_1000.pickle \
  --adv_train ${BASE_PATH}/adv_persample_train_linf_1000.pickle \
  --clean_valid ${BASE_PATH}/clean_valid_1000.pickle \
  --adv_valid ${BASE_PATH}/adv_persample_valid_linf_1000.pickle \
  --clean_jpeg ${BASE_PATH_DEFENSE}/jpeg/ours_linf_clean.pickle \
  --adv_jpeg ${BASE_PATH_DEFENSE}/spatial/ours_linf_persample.pickle \
  --clean_spatial ${BASE_PATH_DEFENSE}/jpeg/ours_linf_clean.pickle \
  --adv_spatial ${BASE_PATH_DEFENSE}/spatial/ours_linf_persample.pickle \
  --model resnet50

BASE_PATH="DeepSloth/complete_datasets/cifar10/cifar10_vgg16bn_sdn_ic_only_1000samples"
BASE_PATH_DEFENSE="DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only"

python train_detector.py \
  --clean_train ${BASE_PATH}/clean_train_1000.pickle \
  --adv_train ${BASE_PATH}/adv_persample_train_l2_1000.pickle \
  --clean_valid ${BASE_PATH}/clean_valid_1000.pickle \
  --adv_valid ${BASE_PATH}/adv_persample_valid_l2_1000.pickle \
  --clean_jpeg ${BASE_PATH_DEFENSE}/jpeg/ours_l2_clean.pickle \
  --adv_jpeg ${BASE_PATH_DEFENSE}/spatial/ours_l2_persample.pickle \
  --clean_spatial ${BASE_PATH_DEFENSE}/jpeg/ours_l2_clean.pickle \
  --adv_spatial ${BASE_PATH_DEFENSE}/spatial/ours_l2_persample.pickle \
  --model resnet50

python train_detector.py \
  --clean_train ${BASE_PATH}/clean_train_1000.pickle \
  --adv_train ${BASE_PATH}/adv_persample_train_linf_1000.pickle \
  --clean_valid ${BASE_PATH}/clean_valid_1000.pickle \
  --adv_valid ${BASE_PATH}/adv_persample_valid_linf_1000.pickle \
  --clean_jpeg ${BASE_PATH_DEFENSE}/jpeg/ours_linf_clean.pickle \
  --adv_jpeg ${BASE_PATH_DEFENSE}/spatial/ours_linf_persample.pickle \
  --clean_spatial ${BASE_PATH_DEFENSE}/jpeg/ours_linf_clean.pickle \
  --adv_spatial ${BASE_PATH_DEFENSE}/spatial/ours_linf_persample.pickle \
  --model resnet50