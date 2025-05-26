#CUDA_VISIBLE_DEVICES=1 python train_sdns.py \
#    --dataset cifar10 \
#    --network resnet56 \
#    --vanilla \
#    --ic-only



# CUDA_VISIBLE_DEVICES=6 python run_attacks.py \
#     --dataset cifar10 \
#     --network resnet56 \
#     --nettype sdn_ic_only \
#     --runmode analysis \
#     --attacks ours \
#     --ellnorm l2 \


# Generate complete datasets with L2 norm
# CUDA_VISIBLE_DEVICES=6 python generate_complete_datasets.py \
#     --dataset cifar10 \
#     --network resnet56 \
#     --nettype sdn_ic_only \
#     --ellnorm l2 \
#     --batch-size 2048 

# CUDA_VISIBLE_DEVICES=6 python generate_complete_datasets.py \
#     --dataset cifar10 \
#     --network resnet56\
#     --nettype sdn_ic_only \
#     --ellnorm linf \
#     --batch-size 2048 

# CUDA_VISIBLE_DEVICES=6 python generate_complete_datasets.py \
#     --dataset cifar10 \
#     --network vgg16bn\
#     --nettype sdn_ic_only \
#     --ellnorm l2 \
#     --batch-size 2048 

CUDA_VISIBLE_DEVICES=6 python generate_complete_datasets.py \
    --dataset cifar10 \
    --network vgg16bn\
    --nettype sdn_ic_only \
    --ellnorm linf \
    --batch-size 2048 

# You can also try other norms:
# --ellnorm l1
# --ellnorm linf