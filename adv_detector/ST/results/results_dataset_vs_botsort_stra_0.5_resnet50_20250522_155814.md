# Adversarial Detector Results

**Generated:** 2025-05-22 15:58:14

## Model Information

- **Feature Extractor:** resnet50
- **Clean Images Folder:** dataset
- **Adversarial Images Folder:** botsort_stra_0.5
- **Train Ratio:** 0.5
- **Validation Size:** 100
- **Batch Size:** 64

## External Test Folders

| Folder Type | Path | Total Images |
|-------------|------|-------------|
| Jpeg | `SlowTrack` | 837 |
| Spatial | `SlowTrack` | 837 |

## Training Results

- **Test Accuracy:** 1.0000
- **Test AUC:** 1.0000
- **Validation Accuracy:** 1.0000
- **Validation AUC:** 1.0000

## Internal Test Split Results

- **Clean Images:** 1.0000 accuracy (175 samples)
- **Adversarial Images:** 1.0000 accuracy (175 samples)
- **Overall:** 1.0000 accuracy

## External Dataset Results

- **Jpeg (Original):** 1.0000 accuracy, 0.0181 avg confidence (837 samples)
- **Spatial (Original):** 1.0000 accuracy, 0.0251 avg confidence (837 samples)

## Detailed Performance Summary

| Dataset | Image Type | Accuracy | Avg Confidence | Sample Count |
|---------|------------|----------|----------------|-------------|
| Internal Test | Clean | 1.0000 | 0.0047 | 175 |
| Internal Test | Adversarial | 1.0000 | 0.9943 | 175 |
| Jpeg | Original | 1.0000 | 0.0181 | 837 |
| Spatial | Original | 1.0000 | 0.0251 | 837 |

## Key Insights

### Training Performance
The detector achieved **100.0%** accuracy on the internal test split, with an AUC of **1.000**, indicating excellent discriminative performance.

### Detection Performance
- **Clean image classification:** 100.0% (lower false positive rate is better)
- **Adversarial detection:** 100.0% (higher detection rate is better)
- **Overall assessment:** Excellent performance on both clean and adversarial images

### Robustness Analysis
- **JPEG robustness:** 100.0% detection rate - detector is robust to JPEG compression
- **Spatial robustness:** 100.0% detection rate - detector is robust to spatial transformations

## Recommendations

- Consider ensemble methods or more advanced feature extractors for improved performance
