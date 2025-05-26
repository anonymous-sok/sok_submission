# Adversarial Detector Results

**Generated:** 2025-05-22 18:40:41

## Model Information

- **Feature Extractor:** resnet50
- **Clean Training File:** clean_train_1000.pickle
- **Adversarial Training File:** adv_persample_train_linf_1000.pickle
- **Clean Validation File:** clean_valid_1000.pickle
- **Adversarial Validation File:** adv_persample_valid_linf_1000.pickle
- **Batch Size:** 32

## Test File Information

| File Type | Clean Path | Adv Path | Exists | Total Images |
|-----------|------------|----------|--------|-------------|
| Validation As Test | `/opt/dlami/nvme/DeepSloth/complete_datasets/cifar10/cifar10_vgg16bn_sdn_ic_only_1000samples/clean_valid_1000.pickle` | `/opt/dlami/nvme/DeepSloth/complete_datasets/cifar10/cifar10_vgg16bn_sdn_ic_only_1000samples/adv_persample_valid_linf_1000.pickle` | ✓/✓ | 2000 |
| Jpeg | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only/jpeg/ours_linf_clean.pickle` | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only/spatial/ours_linf_persample.pickle` | ✓/✓ | 20000 |
| Spatial | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only/jpeg/ours_linf_clean.pickle` | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only/spatial/ours_linf_persample.pickle` | ✓/✓ | 20000 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.7775 | 0.8420 |
| AUC | 0.8778 | 0.9242 |
| Latency | 0.7782s | - |

## Inference Results

### Validation As Test

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.8280 | 0.2372 | Should be classified as clean (0) |
| Adversarial | 0.8560 | 0.7691 | Should be detected as adversarial (1) |
| Overall | 0.8420 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 82.8%
- Adversarial Detection Rate: 85.6%
- False Positive Rate: 17.2%
- False Negative Rate: 14.4%
- Overall Balance: 0.8420
- Inference Time: 3.8588s

### Jpeg

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.8290 | 0.2435 | Should be classified as clean (0) |
| Adversarial | 0.0530 | 0.1109 | Should be detected as adversarial (1) |
| Overall | 0.4410 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 82.9%
- Adversarial Detection Rate: 5.3%
- False Positive Rate: 17.1%
- False Negative Rate: 94.7%
- Overall Balance: 0.4410
- Inference Time: 3.7931s

### Spatial

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.8290 | 0.2435 | Should be classified as clean (0) |
| Adversarial | 0.0530 | 0.1109 | Should be detected as adversarial (1) |
| Overall | 0.4410 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 82.9%
- Adversarial Detection Rate: 5.3%
- False Positive Rate: 17.1%
- False Negative Rate: 94.7%
- Overall Balance: 0.4410
- Inference Time: 3.4991s

## Summary

This detector was trained on clean_train_1000_adv_persample_train_linf_1000 and evaluated on multiple test sets. The model achieved 77.8% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- validation_as_test: 82.8% clean classification, 85.6% adversarial detection
- jpeg: 82.9% clean classification, 5.3% adversarial detection
- spatial: 82.9% clean classification, 5.3% adversarial detection
