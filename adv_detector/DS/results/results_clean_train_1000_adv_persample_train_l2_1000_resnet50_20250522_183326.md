# Adversarial Detector Results

**Generated:** 2025-05-22 18:33:26

## Model Information

- **Feature Extractor:** resnet50
- **Clean Training File:** clean_train_1000.pickle
- **Adversarial Training File:** adv_persample_train_l2_1000.pickle
- **Clean Validation File:** clean_valid_1000.pickle
- **Adversarial Validation File:** adv_persample_valid_l2_1000.pickle
- **Batch Size:** 32

## Test File Information

| File Type | Clean Path | Adv Path | Exists | Total Images |
|-----------|------------|----------|--------|-------------|
| Validation As Test | `/opt/dlami/nvme/DeepSloth/complete_datasets/cifar10/cifar10_resnet56_sdn_ic_only_1000samples/clean_valid_1000.pickle` | `/opt/dlami/nvme/DeepSloth/complete_datasets/cifar10/cifar10_resnet56_sdn_ic_only_1000samples/adv_persample_valid_l2_1000.pickle` | ✓/✓ | 2000 |
| Jpeg | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only/jpeg/ours_l2_clean.pickle` | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only/spatial/ours_l2_persample.pickle` | ✓/✓ | 20000 |
| Spatial | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only/jpeg/ours_l2_clean.pickle` | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only/spatial/ours_l2_persample.pickle` | ✓/✓ | 20000 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.3850 | 0.6235 |
| AUC | 0.6504 | 0.3276 |
| Latency | 1.0464s | - |

## Inference Results

### Validation As Test

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.7070 | 0.5161 | Should be classified as clean (0) |
| Adversarial | 0.5400 | 0.4638 | Should be detected as adversarial (1) |
| Overall | 0.6235 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 70.7%
- Adversarial Detection Rate: 54.0%
- False Positive Rate: 29.3%
- False Negative Rate: 46.0%
- Overall Balance: 0.6235
- Inference Time: 5.3999s

### Jpeg

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.5700 | 0.4855 | Should be classified as clean (0) |
| Adversarial | 0.1920 | 0.5552 | Should be detected as adversarial (1) |
| Overall | 0.3810 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 57.0%
- Adversarial Detection Rate: 19.2%
- False Positive Rate: 43.0%
- False Negative Rate: 80.8%
- Overall Balance: 0.3810
- Inference Time: 5.2753s

### Spatial

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.5700 | 0.4855 | Should be classified as clean (0) |
| Adversarial | 0.1920 | 0.5552 | Should be detected as adversarial (1) |
| Overall | 0.3810 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 57.0%
- Adversarial Detection Rate: 19.2%
- False Positive Rate: 43.0%
- False Negative Rate: 80.8%
- Overall Balance: 0.3810
- Inference Time: 4.7630s

## Summary

This detector was trained on clean_train_1000_adv_persample_train_l2_1000 and evaluated on multiple test sets. The model achieved 38.5% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- validation_as_test: 70.7% clean classification, 54.0% adversarial detection
- jpeg: 57.0% clean classification, 19.2% adversarial detection
- spatial: 57.0% clean classification, 19.2% adversarial detection
