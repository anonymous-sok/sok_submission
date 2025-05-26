CUDA_VISIBLE_DEVICES=3 python train_sdns.py \
    --dataset cifar10 \
    --network mobilenet \
    --vanilla \
    --ic-only