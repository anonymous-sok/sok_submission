# Adversarial Detector Results

**Generated:** 2025-05-22 19:16:06

## Model Information

- **Feature Extractor:** resnet50
- **Training File:** 0_Linf_BEST_flickr8k_googlenet_rnn.adv
- **Validation Size:** 100
- **Batch Size:** 64

## Test File Information

| File Type | Path | Exists | Total Images |
|-----------|------|--------|-------------|
| Test | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/0_Linf_BEST_flickr8k_googlenet_rnn.adv` | ✓ | 2000 |
| Test Jpeg | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/jpeg/0_Linf_BEST_flickr8k_googlenet_rnn.adv` | ✓ | 2000 |
| Test Spatial | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/spatial/0_Linf_BEST_flickr8k_googlenet_rnn.adv` | ✓ | 2000 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.3333 | 0.2100 |
| AUC | 0.7259 | 0.8109 |
| Latency | 0.9695s | - |

## Inference Results

### Test

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.5400 | 0.4752 | 3.0485s |
| Original | 0.6510 | 0.5615 | 3.1396s |

**Detection Summary:**
- Adversarial Detection Rate: 54.0%
- Original Classification Rate: 65.1%
- False Positive Rate: 34.9%
- Overall Balance: 0.5955

### Test Jpeg

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.2140 | 0.6279 | 3.1238s |
| Original | 0.8720 | 0.6863 | 3.0943s |

**Detection Summary:**
- Adversarial Detection Rate: 21.4%
- Original Classification Rate: 87.2%
- False Positive Rate: 12.8%
- Overall Balance: 0.5430

### Test Spatial

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.0190 | 0.8541 | 3.0791s |
| Original | 0.9830 | 0.8578 | 3.0260s |

**Detection Summary:**
- Adversarial Detection Rate: 1.9%
- Original Classification Rate: 98.3%
- False Positive Rate: 1.7%
- Overall Balance: 0.5010

## Summary

This detector was trained on 0_Linf_BEST_flickr8k_googlenet_rnn and evaluated on multiple test sets. The model achieved 33.3% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- test: 54.0% adversarial detection rate
- test_jpeg: 21.4% adversarial detection rate
- test_spatial: 1.9% adversarial detection rate
