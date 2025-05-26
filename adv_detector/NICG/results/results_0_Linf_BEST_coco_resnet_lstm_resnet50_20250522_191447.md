# Adversarial Detector Results

**Generated:** 2025-05-22 19:14:47

## Model Information

- **Feature Extractor:** resnet50
- **Training File:** 0_Linf_BEST_coco_resnet_lstm.adv
- **Validation Size:** 100
- **Batch Size:** 64

## Test File Information

| File Type | Path | Exists | Total Images |
|-----------|------|--------|-------------|
| Test | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/0_Linf_BEST_coco_resnet_lstm.adv` | ✓ | 2200 |
| Test Jpeg | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/jpeg/0_Linf_BEST_coco_resnet_lstm.adv` | ✓ | 2200 |
| Test Spatial | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/spatial/0_Linf_BEST_coco_resnet_lstm.adv` | ✓ | 2200 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.3262 | 0.3200 |
| AUC | 0.7249 | 0.6596 |
| Latency | 1.1497s | - |

## Inference Results

### Test

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.5691 | 0.4524 | 3.7003s |
| Original | 0.6500 | 0.5353 | 3.3601s |

**Detection Summary:**
- Adversarial Detection Rate: 56.9%
- Original Classification Rate: 65.0%
- False Positive Rate: 35.0%
- Overall Balance: 0.6095

### Test Jpeg

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.3464 | 0.5298 | 2.9322s |
| Original | 0.7945 | 0.5917 | 2.9048s |

**Detection Summary:**
- Adversarial Detection Rate: 34.6%
- Original Classification Rate: 79.5%
- False Positive Rate: 20.5%
- Overall Balance: 0.5705

### Test Spatial

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.0500 | 0.7007 | 2.7243s |
| Original | 0.9573 | 0.7039 | 2.7426s |

**Detection Summary:**
- Adversarial Detection Rate: 5.0%
- Original Classification Rate: 95.7%
- False Positive Rate: 4.3%
- Overall Balance: 0.5036

## Summary

This detector was trained on 0_Linf_BEST_coco_resnet_lstm and evaluated on multiple test sets. The model achieved 32.6% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- test: 56.9% adversarial detection rate
- test_jpeg: 34.6% adversarial detection rate
- test_spatial: 5.0% adversarial detection rate
