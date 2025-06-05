# SoK: Efficiency Robustness of Dynamic Deep Learning Systems

Source code implementation for the research paper "SoK: Efficiency Robustness of Dynamic Deep Learning Systems".

## Introduction

This repository contains the implementation code for:
- **Section 5 - Defense for Efficiency Attacks**: Detection strategies (input validation) and mitigation strategies (adversarial training, input transformations) against efficiency attacks
- **Section 6 - Resilience of Existing Defenses**: Comprehensive evaluation of defense effectiveness across MENs (D1), NICG models (D2), and object tracking systems (D3)
- **Experimental framework**: Complete implementation of the evaluation setup including datasets, models, attacks, and metrics used in the empirical analysis

This work addresses the security vulnerabilities in DDLSs that adapt computation based on input complexity, providing both defense implementations and rigorous experimental evaluation of their effectiveness.

**Example Attacks**: We demonstrate our approach using three representative attacks: NICGSlowDown, DeepSloth, and SlowTrack.

## File Structure

```
sok_submission/
├── adv_detector/              # Adversarial attack detection implementations
│   ├── DS/                    # DeepSloth attack detector
│   │   ├── model/             # Trained detector models
│   │   ├── results/           # Training and evaluation results
│   │   ├── run_training.sh    # Training script
│   │   └── train_detector.py  # Detector training code
│   ├── NICG/                  # NICGSlowDown attack detector
│   │   ├── model/             # Trained detector models
│   │   ├── results/           # Training and evaluation results
│   │   ├── run_training.sh    # Training script
│   │   └── train_detector.py  # Detector training code
│   └── ST/                    # SlowTrack attack detector
│       ├── model/             # Trained detector models
│       ├── results/           # Training and evaluation results
│       ├── run_training.sh    # Training script
│       └── train_detector.py  # Detector training code
├── mitigation/                # Mitigation strategies implementation
│   ├── data_loader/           # Dataset loading utilities
│   ├── defense/               # Adversarial training and input transformations
│   ├── img_clean_jpeg/        # Clean images for JPEG compression testing
│   └── img_clean_spatial/     # Clean images for spatial smoothing testing
├── src/                       # Attack implementations and utilities
│   ├── CVPR22_NICGSlowDown/   # NICGSlowDown attack implementation
│   ├── DeepSloth/             # DeepSloth attack implementation
│   ├── SlowTrack/             # SlowTrack attack implementation
│   └── a-PyTorch-Tutorial-to-Image-Captioning/  # NICG input file generation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

```bash
git clone https://github.com/anonymous-sok/sok_submission.git
cd sok_submission
pip install -r requirements.txt
```

## Preparation and Setup

This section walks you through all the necessary preparation steps to set up the experimental environment. You'll need to prepare datasets, train baseline models, and generate adversarial examples before running the main detection and mitigation experiments.

### NICG Attack Dataset and Model Preparation

The NICG Attack require both dataset preparation and model training as foundational steps.

**Dataset Downloads Required:**

You'll need to download several large datasets for the NICG Attack. Please ensure you have sufficient storage space before proceeding:

- [MSCOCO Training Images (13GB)](http://images.cocodataset.org/zips/train2014.zip) - Contains training images for the COCO dataset
- [MSCOCO Validation Images (6GB)](http://images.cocodataset.org/zips/val2014.zip) - Contains validation images for evaluation
- [Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) - Alternative dataset for caption generation
- [Karpathy's Caption Splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) - Contains the standard train/validation/test splits and captions used in research

**Organizing Your Data Directory:**

After downloading all the required files, you need to extract and organize them into the following directory structure. This organization is crucial for the subsequent scripts to locate the data correctly:

```
coco/
├── train2014/
│   ├── COCO_train2014_000000000042.jpg
│   ├── COCO_train2014_000000000043.jpg
│   └── ... (all training images)
├── val2014/
│   ├── COCO_val2014_000000000042.jpg
│   ├── COCO_val2014_000000000043.jpg
│   └── ... (all validation images)
flickr8k/
├── *.jpg (all Flickr8k images)
└── ...
dataset_coco.json      # Caption annotations for COCO
dataset_flickr8k.json  # Caption annotations for Flickr8k
```

**Generating Input Files for Model Training:**

Once you have organized the datasets, you need to preprocess them into the format required by our training pipeline. Navigate to the appropriate directory and run the input file generation script:

```bash
cd src/a-PyTorch-Tutorial-to-Image-Captioning/
python create_input_files.py
```

Make sure to use the following configuration settings when running the script. These parameters have been carefully chosen based on our experimental setup:

- **COCO Dataset Configuration**: `captions_per_image=5, min_word_freq=3`
- **Flickr8k Dataset Configuration**: `captions_per_image=5, min_word_freq=5`

This preprocessing step will create the `coco_5_3/` and `flickr8k_5_5/` directories containing the properly formatted data files needed for training the caption generation models.

**Training NICG Models:**

Before you can run efficiency attacks and defenses, you need to train the baseline neural image caption generation models. These models will serve as the target systems for our attacks. Use the following commands to train the models:

```bash
cd src/CVPR22_NICGSlowDown/
CUDA_VISIBLE_DEVICES=0 python train.py --config=coco_mobilenet_rnn.json
```

You can find various model configuration files in the `src/CVPR22_NICGSlowDown/config/` directory. Each configuration file specifies different model architectures and hyperparameters for comprehensive evaluation.

**Generating Adversarial Examples:**

To generate adversarial examples for the NICG models, execute the attack generation script as described below. For comprehensive implementation details, please consult `src/CVPR22_NICGSlowDown/README.md`.

```bash
cd src/CVPR22_NICGSlowDown/
CUDA_VISIBLE_DEVICES=0 python generate_adv.py --task=0 --attack=0 --norm=0 --split=train
```

The following parameters configure the adversarial generation process:
- **`--task`**: Specifies the target model configuration. Available options include:
  - `0`: `BEST_coco_mobilenet_rnn.pth.tar`
  - `1`: `BEST_coco_resnet_lstm.pth.tar`
  - `2`: `BEST_flickr8k_googlenet_rnn.pth.tar`
  - `3`: `BEST_flickr8k_resnext_lstm.pth.tar`
- **`--attack`**: Defines the attack methodology (default: `0` for NICGSlowDown Attack). Additional attack variants are documented in `src/CVPR22_NICGSlowDown/utils.py`.
- **`--norm`**: Sets the perturbation norm constraint (default: `0` for L2 norm).
- **`--split`**: Determines the dataset partition for adversarial generation. Options: `train`, `val`, or `test`.


### DeepSloth Attack Preparation

The DeepSloth attack targets multi-exit neural networks and requires generating adversarial datasets for evaluation.

**Dataset Requirements:**

Fortunately, for DeepSloth experiments, you don't need to manually download any datasets. The CIFAR-10 dataset will be automatically downloaded when you run the attack generation scripts. This simplifies the setup process significantly.

**Generating DeepSloth Attack Datasets:**

To create the complete adversarial datasets needed for our experiments, you need to run the generation script for different network architectures. Navigate to the DeepSloth directory and execute the following commands:

```bash
cd src/DeepSloth/

# Generate adversarial examples for ResNet56 architecture
python generate_complete_datasets.py \
    --dataset cifar10 \
    --network resnet56 \
    --nettype sdn_ic_only \
    --ellnorm l2 \
    --batch-size 2048

# Generate adversarial examples for VGG16BN architecture  
python generate_complete_datasets.py \
    --dataset cifar10 \
    --network vgg16bn \
    --nettype sdn_ic_only \
    --ellnorm l2 \
    --batch-size 2048
```

These commands will generate comprehensive adversarial datasets for both ResNet56 and VGG16BN network architectures. The generated datasets will be automatically saved to the `src/DeepSloth/complete_datasets/cifar10/` directory for use in subsequent experiments.


**Training DeepSloth models:**

To train the baseline models for DeepSloth, you can use the following commands. These commands will train the ResNet56 and VGG16BN models on the CIFAR-10 dataset. For more details, please refer to the `src/DeepSloth/README.md` file.

```bash
# Train ResNet56 model on CIFAR-10 dataset
CUDA_VISIBLE_DEVICES=1 python train_sdns.py \
   --dataset cifar10 \
   --network resnet56 \    
   --vanilla \
   --ic-only
```

### SlowTrack Attack Preparation

The SlowTrack attack targets object tracking systems and requires generating adversarial perturbations for multi-object tracking scenarios.

**Dataset & Model Availability:**

The SlowTrack experiments use the MOT17 dataset, which has been conveniently included in our repository. You can find the complete dataset already organized in the `src/SlowTrack/dataset/` directory, eliminating the need for manual downloads. For model checkpoints and original dataset, you can find the model and data through [Google Drive](https://drive.google.com/drive/u/0/folders/16dyUawFm3kUTGIr82xPC4p-8yRl5gX08)

**Generating Adversarial Perturbations:**

To create the adversarial examples needed for SlowTrack experiments, you need to run the latency attack generation script. This script will create perturbations designed to increase computational costs in object tracking systems:

```bash
cd src/SlowTrack/

# Generate adversarial perturbations for different confidence thresholds
python tools/latency_attack.py \
    -f exps/example/mot/stra_det_s.py \
    -c bytetrack_s_mot17.pth.tar \
    -b 1 -d 1 --fuse \
    --source=./dataset/ \
    --local_rank=0 \
    --nms=0.45 \
    --conf=0.25  # Also run with --conf=0.5 and --conf=0.75
```

You should run this command three times with different confidence levels (0.25, 0.5, and 0.75) to generate comprehensive experimental datasets. The generated adversarial datasets will be saved to the `src/SlowTrack/botsort_stra/` directory.

## Main Experiments

Now that you have completed all the preparation steps, you can proceed with the main experimental components of our research: adversarial detection and mitigation strategies.

### Mitigation Strategy Experiments

The mitigation experiments evaluate the effectiveness of various defense mechanisms against efficiency attacks. These experiments implement and test the defensive strategies discussed in Section 5 of our paper.

**Setting Up the Mitigation Environment:**

Before running the mitigation experiments, you need to organize your data files in a specific structure that supports both JPEG compression and spatial smoothing defenses. Create the following directory organization:

```
img_clean_jpeg/
├── coco/
│   ├── COCO_val2014_000000000042.jpg
│   ├── COCO_val2014_000000000043.jpg
│   └── ... (all validation images for JPEG testing)
├── SlowTrack/
│   ├── *.jpg (images from MOT17 dataset)
│   └── ... (tracking sequence images)
├── TEST_IMAGES_coco_5_cap_per_img_3_min_word_freq.hdf5
└── TEST_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5

img_clean_spatial/  # Mirror the same structure as img_clean_jpeg/
├── coco/
├── SlowTrack/
├── TEST_IMAGES_coco_5_cap_per_img_3_min_word_freq.hdf5
└── TEST_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5
```

**Running the Mitigation Experiments:**

The mitigation experiments are divided into two main phases: data loading and defense strategy implementation. You need to execute these in the correct order to ensure proper experimental setup:

```bash
cd mitigation/

# Phase 1: Execute all data loading scripts
# These scripts prepare the datasets for defense evaluation
cd data_loader/
# Run all scripts in this directory sequentially
# (Execute each .py file in the directory)

# Phase 2: Execute all defense strategy implementations  
# These scripts test JPEG compression, spatial smoothing, and adversarial training
cd ../defense/
# Run all scripts in this directory sequentially
# (Execute each .py file in the directory)
```

The mitigation experiments will generate processed datasets that incorporate various defensive transformations. These processed datasets will then be used in the detector evaluation phase to assess how well defenses can maintain their effectiveness under adversarial conditions.

### Adversarial Detector Training and Evaluation

This section covers the training and evaluation of machine learning models designed to detect efficiency attacks. These detectors implement the input validation strategies discussed in our research and represent a key component of our defensive framework.

**Training Individual Attack Detectors:**

We provide specialized detector training for each of the three attack types studied in our research. Each detector is trained to identify the specific patterns and characteristics of its target attack. Execute the following commands to train each detector:

```bash
# Training the DeepSloth Attack Detector
# This detector learns to identify efficiency attacks on multi-exit networks
cd adv_detector/DS/
chmod +x run_training.sh
./run_training.sh

# Training the NICG (Neural Image Caption Generation) Attack Detector  
# This detector specializes in identifying attacks on caption generation systems
cd ../NICG/
chmod +x run_training.sh
./run_training.sh

# Training the SlowTrack Attack Detector
# This detector focuses on identifying attacks against object tracking systems
cd ../ST/
chmod +x run_training.sh
./run_training.sh
```

**Training Output:**

Each detector training process will generate several outputs:

- **Trained Models**: The final trained detector models will be saved in the respective `./model/` directories within each attack-specific folder
- **Training Logs**: Comprehensive training logs documenting the learning process, loss curves, and performance metrics will be stored in the `./results/` directories
- **Evaluation Results**: Detailed evaluation results, including accuracy metrics, detection rates, and performance analysis, can be found in the `./results/results_*.md` files within each detector directory

**Interpreting the Results:**

All experimental results from the detector training and evaluation have been carefully analyzed and summarized in our research paper. For a comprehensive understanding of the detection performance and comparative analysis across different attack types, please refer to **Table 4** in the paper, which presents the consolidated results from all detector experiments.

## Important Notes

- **Path Configuration**: Update all file paths in scripts according to your directory structure
- **Results**: All experimental results are summarized in the paper's Table 4
- **File Size**: Some source files excluded due to size limitations - follow dataset preparation steps to generate required files
