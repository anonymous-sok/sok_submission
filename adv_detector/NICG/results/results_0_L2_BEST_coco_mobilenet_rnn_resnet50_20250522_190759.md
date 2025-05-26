# Adversarial Detector Results

**Generated:** 2025-05-22 19:07:59

## Model Information

- **Feature Extractor:** resnet50
- **Training File:** 0_L2_BEST_coco_mobilenet_rnn.adv
- **Validation Size:** 100
- **Batch Size:** 64

## Test File Information

| File Type | Path | Exists | Total Images |
|-----------|------|--------|-------------|
| Test | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/0_L2_BEST_coco_mobilenet_rnn.adv` | ✓ | 2200 |
| Test Jpeg | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/jpeg/0_L2_BEST_coco_mobilenet_rnn.adv` | ✓ | 2200 |
| Test Spatial | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/spatial/0_L2_BEST_coco_mobilenet_rnn.adv` | ✓ | 2200 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.3571 | 0.3900 |
| AUC | 0.6636 | 0.6364 |
| Latency | 1.1474s | - |

## Inference Results

### Test

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.6445 | 0.4525 | 3.1164s |
| Original | 0.6455 | 0.5289 | 3.0714s |

**Detection Summary:**
- Adversarial Detection Rate: 64.5%
- Original Classification Rate: 64.5%
- False Positive Rate: 35.5%
- Overall Balance: 0.6450

### Test Jpeg

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.2909 | 0.5508 | 2.7672s |
| Original | 0.8727 | 0.6072 | 3.2775s |

**Detection Summary:**
- Adversarial Detection Rate: 29.1%
- Original Classification Rate: 87.3%
- False Positive Rate: 12.7%
- Overall Balance: 0.5818

### Test Spatial

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.0282 | 0.6975 | 2.9281s |
| Original | 0.9718 | 0.7006 | 2.4407s |

**Detection Summary:**
- Adversarial Detection Rate: 2.8%
- Original Classification Rate: 97.2%
- False Positive Rate: 2.8%
- Overall Balance: 0.5000

## Summary

This detector was trained on 0_L2_BEST_coco_mobilenet_rnn and evaluated on multiple test sets. The model achieved 35.7% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- test: 64.5% adversarial detection rate
- test_jpeg: 29.1% adversarial detection rate
- test_spatial: 2.8% adversarial detection rate
