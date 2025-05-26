# Adversarial Detector Results

**Generated:** 2025-05-22 19:09:26

## Model Information

- **Feature Extractor:** resnet50
- **Training File:** 0_L2_BEST_coco_resnet_lstm.adv
- **Validation Size:** 100
- **Batch Size:** 64

## Test File Information

| File Type | Path | Exists | Total Images |
|-----------|------|--------|-------------|
| Test | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/0_L2_BEST_coco_resnet_lstm.adv` | ✓ | 2200 |
| Test Jpeg | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/jpeg/0_L2_BEST_coco_resnet_lstm.adv` | ✓ | 2200 |
| Test Spatial | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/spatial/0_L2_BEST_coco_resnet_lstm.adv` | ✓ | 2200 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.4429 | 0.4800 |
| AUC | 0.5172 | 0.4505 |
| Latency | 1.0349s | - |

## Inference Results

### Test

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.6909 | 0.4952 | 2.6886s |
| Original | 0.6745 | 0.5003 | 2.9108s |

**Detection Summary:**
- Adversarial Detection Rate: 69.1%
- Original Classification Rate: 67.5%
- False Positive Rate: 32.5%
- Overall Balance: 0.6827

### Test Jpeg

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.4064 | 0.4991 | 2.9785s |
| Original | 0.8264 | 0.5028 | 2.8857s |

**Detection Summary:**
- Adversarial Detection Rate: 40.6%
- Original Classification Rate: 82.6%
- False Positive Rate: 17.4%
- Overall Balance: 0.6164

### Test Spatial

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.0336 | 0.5085 | 2.6041s |
| Original | 0.9700 | 0.5089 | 2.8665s |

**Detection Summary:**
- Adversarial Detection Rate: 3.4%
- Original Classification Rate: 97.0%
- False Positive Rate: 3.0%
- Overall Balance: 0.5018

## Summary

This detector was trained on 0_L2_BEST_coco_resnet_lstm and evaluated on multiple test sets. The model achieved 44.3% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- test: 69.1% adversarial detection rate
- test_jpeg: 40.6% adversarial detection rate
- test_spatial: 3.4% adversarial detection rate
