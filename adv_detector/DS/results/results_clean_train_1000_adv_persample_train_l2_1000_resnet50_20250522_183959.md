# Adversarial Detector Results

**Generated:** 2025-05-22 18:39:59

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
| Validation As Test | `/opt/dlami/nvme/DeepSloth/complete_datasets/cifar10/cifar10_vgg16bn_sdn_ic_only_1000samples/clean_valid_1000.pickle` | `/opt/dlami/nvme/DeepSloth/complete_datasets/cifar10/cifar10_vgg16bn_sdn_ic_only_1000samples/adv_persample_valid_l2_1000.pickle` | ✓/✓ | 2000 |
| Jpeg | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only/jpeg/ours_l2_clean.pickle` | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only/spatial/ours_l2_persample.pickle` | ✓/✓ | 20000 |
| Spatial | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only/jpeg/ours_l2_clean.pickle` | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only/spatial/ours_l2_persample.pickle` | ✓/✓ | 20000 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.3575 | 0.6025 |
| AUC | 0.6795 | 0.3526 |
| Latency | 1.1124s | - |

## Inference Results

### Validation As Test

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.6820 | 0.5210 | Should be classified as clean (0) |
| Adversarial | 0.5230 | 0.4668 | Should be detected as adversarial (1) |
| Overall | 0.6025 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 68.2%
- Adversarial Detection Rate: 52.3%
- False Positive Rate: 31.8%
- False Negative Rate: 47.7%
- Overall Balance: 0.6025
- Inference Time: 5.0605s

### Jpeg

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.5200 | 0.4789 | Should be classified as clean (0) |
| Adversarial | 0.2980 | 0.5265 | Should be detected as adversarial (1) |
| Overall | 0.4090 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 52.0%
- Adversarial Detection Rate: 29.8%
- False Positive Rate: 48.0%
- False Negative Rate: 70.2%
- Overall Balance: 0.4090
- Inference Time: 5.5418s

### Spatial

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.5200 | 0.4789 | Should be classified as clean (0) |
| Adversarial | 0.2980 | 0.5265 | Should be detected as adversarial (1) |
| Overall | 0.4090 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 52.0%
- Adversarial Detection Rate: 29.8%
- False Positive Rate: 48.0%
- False Negative Rate: 70.2%
- Overall Balance: 0.4090
- Inference Time: 5.1046s

## Summary

This detector was trained on clean_train_1000_adv_persample_train_l2_1000 and evaluated on multiple test sets. The model achieved 35.8% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- validation_as_test: 68.2% clean classification, 52.3% adversarial detection
- jpeg: 52.0% clean classification, 29.8% adversarial detection
- spatial: 52.0% clean classification, 29.8% adversarial detection
