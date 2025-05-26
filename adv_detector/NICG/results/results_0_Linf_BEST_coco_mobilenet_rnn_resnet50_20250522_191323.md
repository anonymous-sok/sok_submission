# Adversarial Detector Results

**Generated:** 2025-05-22 19:13:23

## Model Information

- **Feature Extractor:** resnet50
- **Training File:** 0_Linf_BEST_coco_mobilenet_rnn.adv
- **Validation Size:** 100
- **Batch Size:** 64

## Test File Information

| File Type | Path | Exists | Total Images |
|-----------|------|--------|-------------|
| Test | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/0_Linf_BEST_coco_mobilenet_rnn.adv` | ✓ | 2200 |
| Test Jpeg | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/jpeg/0_Linf_BEST_coco_mobilenet_rnn.adv` | ✓ | 2200 |
| Test Spatial | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/spatial/0_Linf_BEST_coco_mobilenet_rnn.adv` | ✓ | 2200 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.3286 | 0.3700 |
| AUC | 0.7079 | 0.6594 |
| Latency | 1.0709s | - |

## Inference Results

### Test

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.6036 | 0.4484 | 2.9894s |
| Original | 0.6464 | 0.5354 | 2.5604s |

**Detection Summary:**
- Adversarial Detection Rate: 60.4%
- Original Classification Rate: 64.6%
- False Positive Rate: 35.4%
- Overall Balance: 0.6250

### Test Jpeg

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.2745 | 0.5699 | 2.5547s |
| Original | 0.8673 | 0.6333 | 2.8232s |

**Detection Summary:**
- Adversarial Detection Rate: 27.5%
- Original Classification Rate: 86.7%
- False Positive Rate: 13.3%
- Overall Balance: 0.5709

### Test Spatial

**Sample Counts:** 1100 adversarial, 1100 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.0355 | 0.7385 | 2.6314s |
| Original | 0.9645 | 0.7415 | 2.6495s |

**Detection Summary:**
- Adversarial Detection Rate: 3.5%
- Original Classification Rate: 96.5%
- False Positive Rate: 3.5%
- Overall Balance: 0.5000

## Summary

This detector was trained on 0_Linf_BEST_coco_mobilenet_rnn and evaluated on multiple test sets. The model achieved 32.9% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- test: 60.4% adversarial detection rate
- test_jpeg: 27.5% adversarial detection rate
- test_spatial: 3.5% adversarial detection rate
