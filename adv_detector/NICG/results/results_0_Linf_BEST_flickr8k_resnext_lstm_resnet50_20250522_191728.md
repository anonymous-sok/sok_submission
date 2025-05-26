# Adversarial Detector Results

**Generated:** 2025-05-22 19:17:28

## Model Information

- **Feature Extractor:** resnet50
- **Training File:** 0_Linf_BEST_flickr8k_resnext_lstm.adv
- **Validation Size:** 100
- **Batch Size:** 64

## Test File Information

| File Type | Path | Exists | Total Images |
|-----------|------|--------|-------------|
| Test | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/0_Linf_BEST_flickr8k_resnext_lstm.adv` | ✓ | 2000 |
| Test Jpeg | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/jpeg/0_Linf_BEST_flickr8k_resnext_lstm.adv` | ✓ | 2000 |
| Test Spatial | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/spatial/0_Linf_BEST_flickr8k_resnext_lstm.adv` | ✓ | 2000 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.2929 | 0.2400 |
| AUC | 0.7734 | 0.8440 |
| Latency | 0.9725s | - |

## Inference Results

### Test

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.5140 | 0.4817 | 2.7571s |
| Original | 0.6390 | 0.5579 | 2.7965s |

**Detection Summary:**
- Adversarial Detection Rate: 51.4%
- Original Classification Rate: 63.9%
- False Positive Rate: 36.1%
- Overall Balance: 0.5765

### Test Jpeg

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.2470 | 0.6207 | 3.5319s |
| Original | 0.8240 | 0.6722 | 2.7003s |

**Detection Summary:**
- Adversarial Detection Rate: 24.7%
- Original Classification Rate: 82.4%
- False Positive Rate: 17.6%
- Overall Balance: 0.5355

### Test Spatial

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.0260 | 0.8239 | 3.1377s |
| Original | 0.9760 | 0.8292 | 2.9499s |

**Detection Summary:**
- Adversarial Detection Rate: 2.6%
- Original Classification Rate: 97.6%
- False Positive Rate: 2.4%
- Overall Balance: 0.5010

## Summary

This detector was trained on 0_Linf_BEST_flickr8k_resnext_lstm and evaluated on multiple test sets. The model achieved 29.3% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- test: 51.4% adversarial detection rate
- test_jpeg: 24.7% adversarial detection rate
- test_spatial: 2.6% adversarial detection rate
