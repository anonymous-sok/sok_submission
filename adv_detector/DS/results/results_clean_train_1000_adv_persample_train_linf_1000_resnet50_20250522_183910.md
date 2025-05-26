# Adversarial Detector Results

**Generated:** 2025-05-22 18:39:10

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
| Validation As Test | `/opt/dlami/nvme/DeepSloth/complete_datasets/cifar10/cifar10_resnet56_sdn_ic_only_1000samples/clean_valid_1000.pickle` | `/opt/dlami/nvme/DeepSloth/complete_datasets/cifar10/cifar10_resnet56_sdn_ic_only_1000samples/adv_persample_valid_linf_1000.pickle` | ✓/✓ | 2000 |
| Jpeg | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only/jpeg/ours_linf_clean.pickle` | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only/spatial/ours_linf_persample.pickle` | ✓/✓ | 20000 |
| Spatial | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only/jpeg/ours_linf_clean.pickle` | `/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only/spatial/ours_linf_persample.pickle` | ✓/✓ | 20000 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.7750 | 0.8270 |
| AUC | 0.8662 | 0.9101 |
| Latency | 0.7288s | - |

## Inference Results

### Validation As Test

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.7980 | 0.2624 | Should be classified as clean (0) |
| Adversarial | 0.8560 | 0.7511 | Should be detected as adversarial (1) |
| Overall | 0.8270 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 79.8%
- Adversarial Detection Rate: 85.6%
- False Positive Rate: 20.2%
- False Negative Rate: 14.4%
- Overall Balance: 0.8270
- Inference Time: 3.9486s

### Jpeg

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.8100 | 0.2621 | Should be classified as clean (0) |
| Adversarial | 0.0320 | 0.0832 | Should be detected as adversarial (1) |
| Overall | 0.4210 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 81.0%
- Adversarial Detection Rate: 3.2%
- False Positive Rate: 19.0%
- False Negative Rate: 96.8%
- Overall Balance: 0.4210
- Inference Time: 3.6609s

### Spatial

**Sample Counts:** 1000 clean, 1000 adversarial

| Image Type | Accuracy | Avg Confidence | Details |
|------------|----------|----------------|----------|
| Clean | 0.8100 | 0.2621 | Should be classified as clean (0) |
| Adversarial | 0.0320 | 0.0832 | Should be detected as adversarial (1) |
| Overall | 0.4210 | - | Combined accuracy |

**Detection Summary:**
- Clean Classification Rate: 81.0%
- Adversarial Detection Rate: 3.2%
- False Positive Rate: 19.0%
- False Negative Rate: 96.8%
- Overall Balance: 0.4210
- Inference Time: 3.8378s

## Summary

This detector was trained on clean_train_1000_adv_persample_train_linf_1000 and evaluated on multiple test sets. The model achieved 77.5% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- validation_as_test: 79.8% clean classification, 85.6% adversarial detection
- jpeg: 81.0% clean classification, 3.2% adversarial detection
- spatial: 81.0% clean classification, 3.2% adversarial detection
