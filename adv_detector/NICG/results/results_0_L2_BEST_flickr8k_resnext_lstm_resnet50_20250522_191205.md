# Adversarial Detector Results

**Generated:** 2025-05-22 19:12:05

## Model Information

- **Feature Extractor:** resnet50
- **Training File:** 0_L2_BEST_flickr8k_resnext_lstm.adv
- **Validation Size:** 100
- **Batch Size:** 64

## Test File Information

| File Type | Path | Exists | Total Images |
|-----------|------|--------|-------------|
| Test | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/0_L2_BEST_flickr8k_resnext_lstm.adv` | ✓ | 2000 |
| Test Jpeg | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/jpeg/0_L2_BEST_flickr8k_resnext_lstm.adv` | ✓ | 2000 |
| Test Spatial | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/spatial/0_L2_BEST_flickr8k_resnext_lstm.adv` | ✓ | 2000 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.5333 | 0.4200 |
| AUC | 0.5325 | 0.5253 |
| Latency | 0.9238s | - |

## Inference Results

### Test

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.6450 | 0.4999 | 2.4587s |
| Original | 0.7570 | 0.4989 | 3.1984s |

**Detection Summary:**
- Adversarial Detection Rate: 64.5%
- Original Classification Rate: 75.7%
- False Positive Rate: 24.3%
- Overall Balance: 0.7010

### Test Jpeg

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.3100 | 0.4992 | 2.8966s |
| Original | 0.9110 | 0.4975 | 3.1663s |

**Detection Summary:**
- Adversarial Detection Rate: 31.0%
- Original Classification Rate: 91.1%
- False Positive Rate: 8.9%
- Overall Balance: 0.6105

### Test Spatial

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.0100 | 0.4941 | 2.7245s |
| Original | 0.9900 | 0.4936 | 2.7160s |

**Detection Summary:**
- Adversarial Detection Rate: 1.0%
- Original Classification Rate: 99.0%
- False Positive Rate: 1.0%
- Overall Balance: 0.5000

## Summary

This detector was trained on 0_L2_BEST_flickr8k_resnext_lstm and evaluated on multiple test sets. The model achieved 53.3% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- test: 64.5% adversarial detection rate
- test_jpeg: 31.0% adversarial detection rate
- test_spatial: 1.0% adversarial detection rate
