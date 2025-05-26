# Adversarial Detector Results

**Generated:** 2025-05-22 19:10:46

## Model Information

- **Feature Extractor:** resnet50
- **Training File:** 0_L2_BEST_flickr8k_googlenet_rnn.adv
- **Validation Size:** 100
- **Batch Size:** 64

## Test File Information

| File Type | Path | Exists | Total Images |
|-----------|------|--------|-------------|
| Test | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/0_L2_BEST_flickr8k_googlenet_rnn.adv` | ✓ | 2000 |
| Test Jpeg | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/jpeg/0_L2_BEST_flickr8k_googlenet_rnn.adv` | ✓ | 2000 |
| Test Spatial | `/opt/dlami/nvme/CVPR22_NICGSlowDown/adv/spatial/0_L2_BEST_flickr8k_googlenet_rnn.adv` | ✓ | 2000 |

## Training Results

| Metric | Test | Validation |
|--------|------|------------|
| Accuracy | 0.4976 | 0.4300 |
| AUC | 0.5458 | 0.5030 |
| Latency | 1.0736s | - |

## Inference Results

### Test

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.6830 | 0.5005 | 2.3262s |
| Original | 0.7170 | 0.4960 | 2.4483s |

**Detection Summary:**
- Adversarial Detection Rate: 68.3%
- Original Classification Rate: 71.7%
- False Positive Rate: 28.3%
- Overall Balance: 0.7000

### Test Jpeg

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.2670 | 0.4959 | 2.2642s |
| Original | 0.9200 | 0.4919 | 2.6282s |

**Detection Summary:**
- Adversarial Detection Rate: 26.7%
- Original Classification Rate: 92.0%
- False Positive Rate: 8.0%
- Overall Balance: 0.5935

### Test Spatial

**Sample Counts:** 1000 adversarial, 1000 original

| Image Type | Accuracy | Avg Confidence | Inference Time |
|------------|----------|----------------|----------------|
| Adversarial | 0.0060 | 0.4840 | 2.6422s |
| Original | 0.9960 | 0.4835 | 2.6906s |

**Detection Summary:**
- Adversarial Detection Rate: 0.6%
- Original Classification Rate: 99.6%
- False Positive Rate: 0.4%
- Overall Balance: 0.5010

## Summary

This detector was trained on 0_L2_BEST_flickr8k_googlenet_rnn and evaluated on multiple test sets. The model achieved 49.8% test accuracy during training and showed varying performance across different attack types during inference.

**Key Findings:**
- test: 68.3% adversarial detection rate
- test_jpeg: 26.7% adversarial detection rate
- test_spatial: 0.6% adversarial detection rate
