#CUDA_VISIBLE_DEVICES=2 python train_sdns.py \
#    --dataset cifar10 \
#    --network vgg16bn \
#    --vanilla \
#    --ic-only


CUDA_VISIBLE_DEVICES=7 python run_attacks.py \
    --dataset cifar10 \
    --network vgg16bn \
    --nettype sdn_ic_only \
    --runmode analysis \
    --attacks ours \
    --ellnorm l2 \