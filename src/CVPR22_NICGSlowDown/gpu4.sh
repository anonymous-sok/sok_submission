
#CUDA_VISIBLE_DEVICES=0 python train.py --config=coco_mobilenet_rnn.json

CUDA_VISIBLE_DEVICES=0 python generate_adv.py --task=0 --attack=0 --norm=0 --split=train &
CUDA_VISIBLE_DEVICES=1 python generate_adv.py --task=0 --attack=0 --norm=1 --split=train &

CUDA_VISIBLE_DEVICES=2 python generate_adv.py --task=1 --attack=0 --norm=0 --split=train &
CUDA_VISIBLE_DEVICES=3 python generate_adv.py --task=1 --attack=0 --norm=1 --split=train &

CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=2 --attack=0 --norm=0 --split=train &
CUDA_VISIBLE_DEVICES=5 python generate_adv.py --task=2 --attack=0 --norm=1 --split=train &

CUDA_VISIBLE_DEVICES=6 python generate_adv.py --task=3 --attack=0 --norm=0 --split=train &
CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=0 --norm=1 --split=train

# CUDA_VISIBLE_DEVICES=4 python test_latency.py --task=0 --attack=0
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=1 --norm=0
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=1 --norm=1

#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=2 --norm=0
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=2 --norm=1

#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=3 --norm=0
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=3 --norm=1
#
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=4 --norm=0
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=4 --norm=1
#
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=5 --norm=0
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=5 --norm=1
#
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=6 --norm=0
#CUDA_VISIBLE_DEVICES=4 python generate_adv.py --task=0 --attack=6 --norm=1

#CUDA_VISIBLE_DEVICES=4 python test_latency.py --task=0 --attack=1
#CUDA_VISIBLE_DEVICES=4 python test_latency.py --task=0 --attack=2
#CUDA_VISIBLE_DEVICES=4 python test_latency.py --task=0 --attack=3
#CUDA_VISIBLE_DEVICES=4 python test_latency.py --task=0 --attack=4
#CUDA_VISIBLE_DEVICES=4 python test_latency.py --task=0 --attack=5
#CUDA_VISIBLE_DEVICES=4 python test_latency.py --task=0 --attack=6


# CUDA_VISIBLE_DEVICES=4 python loss_impact.py --task=0