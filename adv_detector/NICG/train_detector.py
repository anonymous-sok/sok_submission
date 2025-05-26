import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn import metrics
import pyRAPL
from PIL import Image
import argparse
from tqdm import tqdm


model_dir = 'model'
results_dir = 'results'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdvImageDataset(Dataset):
    """Dataset class for loading adversarial and benign images from .adv files"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert tensor to PIL for transforms if needed
        if self.transform:
            # Ensure image is in [0,1] range and convert to PIL
            if image.max() <= 1.0:
                image_pil = transforms.ToPILImage()(image)
            else:
                image_pil = transforms.ToPILImage()(image / 255.0)
            image = self.transform(image_pil)
        
        return image, label

class FeatureExtractor(nn.Module):
    """Feature extractor using pre-trained models"""
    def __init__(self, model_name='resnet50'):
        super().__init__()
        if model_name == 'resnet50':
            import torchvision.models as models
            self.backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove classifier
            self.feature_dim = 2048
        elif model_name == 'vit':
            # For Vision Transformer
            try:
                import timm
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
                self.feature_dim = 768
            except ImportError:
                print("timm not available, falling back to ResNet50")
                import torchvision.models as models
                self.backbone = models.resnet50(pretrained=True)
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
                self.feature_dim = 2048
        
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
    
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)  # Flatten
            return features

def load_adv_data(adv_file_path):
    """
    Load adversarial data from .adv file
    
    Args:
        adv_file_path: Path to .adv file
    
    Returns:
        images: List of image tensors
        labels: List of labels (0=original, 1=adversarial)
    """
    print(f"Loading data from: {adv_file_path}")
    
    # Load the .adv file - use device for GPU loading if available and file is small enough
    # For large files, it's safer to load to CPU first to avoid GPU memory issues
    try:
        # Try loading to GPU first if available
        if torch.cuda.is_available():
            data = torch.load(adv_file_path, map_location=device, weights_only=False)
        else:
            data = torch.load(adv_file_path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory insufficient, loading to CPU instead...")
            data = torch.load(adv_file_path, map_location='cpu', weights_only=False)
        else:
            raise e
    
    all_images = []
    all_labels = []
    
    print("Processing samples...")
    for sample in tqdm(data, desc="Processing samples"):
        ori_img_list, adv_img_list = sample[0], sample[1]
        
        # Extract image tensors (first element in each list)
        ori_images = ori_img_list[0]  # Shape: [100, 3, 256, 256]
        adv_images = adv_img_list[0]  # Shape: [100, 3, 256, 256]
        
        # Add individual images and labels
        for i in range(ori_images.shape[0]):
            all_images.append(ori_images[i])  # Single image: [3, 256, 256]
            all_labels.append(0)  # Original = 0
            
            all_images.append(adv_images[i])  # Single image: [3, 256, 256]
            all_labels.append(1)  # Adversarial = 1
    
    print(f"Loaded {len(all_images)} images ({len(all_images)//2} pairs)")
    print(f"Original images: {all_labels.count(0)}")
    print(f"Adversarial images: {all_labels.count(1)}")
    
    return all_images, all_labels

def extract_features(dataset, feature_extractor, batch_size=32):
    """Extract features from images using the feature extractor"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_features = []
    all_labels = []
    
    feature_extractor.eval()
    feature_extractor.to(device)
    
    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Feature extraction"):
            images = images.to(device)
            features = feature_extractor(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Feature extraction complete. Shape: {features.shape}")
    return features, labels

def load_adv_data_inference(adv_file_path):
    """
    Load adversarial data from .adv file for inference (only adversarial images)
    
    Args:
        adv_file_path: Path to .adv file
    
    Returns:
        images: List of adversarial image tensors only
        labels: List of labels (all 1s for adversarial)
    """
    print(f"Loading inference data from: {adv_file_path}")
    
    # Load the .adv file
    try:
        if torch.cuda.is_available():
            data = torch.load(adv_file_path, map_location=device, weights_only=False)
        else:
            data = torch.load(adv_file_path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory insufficient, loading to CPU instead...")
            data = torch.load(adv_file_path, map_location='cpu', weights_only=False)
        else:
            raise e
    
    all_images = []
    all_labels = []
    
    print("Processing adversarial samples for inference...")
    for sample in tqdm(data, desc="Processing samples"):
        ori_img_list, adv_img_list = sample[0], sample[1]
        
        # Extract only adversarial image tensors (second element in each list)
        adv_images = adv_img_list[0]  # Shape: [100, 3, 256, 256]
        
        # Add only adversarial images
        for i in range(adv_images.shape[0]):
            all_images.append(adv_images[i])  # Single image: [3, 256, 256]
            all_labels.append(1)  # All adversarial = 1
    
    print(f"Loaded {len(all_images)} adversarial images for inference")
    
    return all_images, all_labels

def load_original_data_inference(adv_file_path):
    """
    Load original data from .adv file for inference (only original images)
    
    Args:
        adv_file_path: Path to .adv file
    
    Returns:
        images: List of original image tensors only
        labels: List of labels (all 0s for original)
    """
    print(f"Loading original inference data from: {adv_file_path}")
    
    # Load the .adv file
    try:
        if torch.cuda.is_available():
            data = torch.load(adv_file_path, map_location=device, weights_only=False)
        else:
            data = torch.load(adv_file_path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory insufficient, loading to CPU instead...")
            data = torch.load(adv_file_path, map_location='cpu', weights_only=False)
        else:
            raise e
    
    all_images = []
    all_labels = []
    
    print("Processing original samples for inference...")
    for sample in tqdm(data, desc="Processing samples"):
        ori_img_list, adv_img_list = sample[0], sample[1]
        
        # Extract only original image tensors (first element in each list)
        ori_images = ori_img_list[0]  # Shape: [100, 3, 256, 256]
        
        # Add only original images
        for i in range(ori_images.shape[0]):
            all_images.append(ori_images[i])  # Single image: [3, 256, 256]
            all_labels.append(0)  # All original = 0
    
    print(f"Loaded {len(all_images)} original images for inference")
    
    return all_images, all_labels

def get_file_basename(file_path):
    """Extract basename from file path for naming saved files"""
    return os.path.splitext(os.path.basename(file_path))[0]

def train_detector(adv_file_path, model_name='resnet50', valid_size=100, batch_size=32):
    """
    Train a detector for a specific attack type
    
    Args:
        adv_file_path: Path to .adv file
        model_name: Feature extractor model ('resnet50' or 'vit')
        valid_size: Number of samples for validation
        batch_size: Batch size for feature extraction
    """
    file_basename = get_file_basename(adv_file_path)
    
    print(f"\n{'='*60}")
    print(f"Training detector for {file_basename} using {model_name}")
    print(f"{'='*60}")
    
    # Load data
    images, labels = load_adv_data(adv_file_path)
    
    # Create dataset with transforms for feature extraction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AdvImageDataset(images, labels, transform=transform)
    
    # Extract features
    feature_extractor = FeatureExtractor(model_name)
    features, labels = extract_features(dataset, feature_extractor, batch_size=batch_size)
    
    # Split data: validation + train/test split
    n_samples = len(features)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    # First take out validation set
    valid_indices = indices[:valid_size]
    remaining_indices = indices[valid_size:]
    
    # Then split remaining into train/test (80/20)
    n_remaining = len(remaining_indices)
    test_size = int(n_remaining * 0.2)
    test_indices = remaining_indices[:test_size]
    train_indices = remaining_indices[test_size:]
    
    # Create splits
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    valid_features = features[valid_indices]
    valid_labels = labels[valid_indices]
    
    print(f"Data split:")
    print(f"  Training: {len(train_features)} samples")
    print(f"  Testing: {len(test_features)} samples")
    print(f"  Validation: {len(valid_features)} samples")
    
    # Train SVM
    print("Training SVM classifier...")
    svm_model = SVC(probability=True, kernel='rbf', random_state=42, verbose=True)
    
    # Create a simple progress bar for SVM training
    print("  Starting SVM optimization...")
    start_time = time.time()
    svm_model.fit(train_features, train_labels)
    end_time = time.time()
    print(f"  SVM training completed in {end_time - start_time:.2f} seconds")
    
    # Save model and data
    save_name = os.path.join(model_dir, f'detector_{file_basename}_{model_name}.pkl')
    torch.save({
        'model': svm_model,
        'train_data': (train_features, train_labels),
        'test_data': (test_features, test_labels),
        'valid_data': (valid_features, valid_labels),
        'feature_extractor': model_name,
        'source_file': adv_file_path
    }, save_name)
    
    # Evaluate on test set
    print("Evaluating detector on test set...")
    
    # Measure inference time
    t1 = time.time()
    test_pred = svm_model.predict(test_features)
    test_proba = svm_model.predict_proba(test_features)[:, 1]
    t2 = time.time()
    
    latency = t2 - t1
    energy = 0  # Placeholder if energy measurement not available
    
    # Calculate metrics
    test_acc = np.mean(test_pred == test_labels)
    fpr, tpr, _ = metrics.roc_curve(test_labels, test_proba, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    
    # Find optimal threshold using validation set
    valid_pred = svm_model.predict(valid_features)
    valid_proba = svm_model.predict_proba(valid_features)[:, 1]
    valid_acc = np.mean(valid_pred == valid_labels)
    fpr_v, tpr_v, thresholds_v = metrics.roc_curve(valid_labels, valid_proba, pos_label=1)
    valid_auc = metrics.auc(fpr_v, tpr_v)
    
    # Find best threshold that maximizes accuracy on validation set
    best_threshold = 0.5
    best_acc = valid_acc
    for threshold in thresholds_v:
        valid_pred_thresh = (valid_proba >= threshold).astype(int)
        acc_thresh = np.mean(valid_pred_thresh == valid_labels)
        if acc_thresh > best_acc:
            best_acc = acc_thresh
            best_threshold = threshold
    
    # Apply best threshold to test set
    test_pred_optimal = (test_proba >= best_threshold).astype(int)
    test_acc_optimal = np.mean(test_pred_optimal == test_labels)
    
    # Check class distribution
    test_class_dist = np.bincount(test_labels)
    valid_class_dist = np.bincount(valid_labels)
    
    print(f"\nResults for {file_basename}:")
    print(f"  Class distribution - Test: {test_class_dist}, Valid: {valid_class_dist}")
    print(f"  Default threshold (0.5):")
    print(f"    Test  - Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
    print(f"    Valid - Accuracy: {valid_acc:.4f}, AUC: {valid_auc:.4f}")
    print(f"  Optimal threshold ({best_threshold:.3f}):")
    print(f"    Test  - Accuracy: {test_acc_optimal:.4f}")
    print(f"    Valid - Accuracy: {best_acc:.4f}")
    print(f"  Latency: {latency:.4f}s, Energy: {energy}")
    
    # Save results
    results = [[file_basename, test_acc, test_auc, valid_acc, valid_auc, latency, energy]]
    os.makedirs(results_dir, exist_ok=True)
    np.savetxt(f'{results_dir}/detector_results_{file_basename}_{model_name}.csv', 
               results, delimiter=',', fmt='%s',
               header='attack_id,test_acc,test_auc,valid_acc,valid_auc,latency,energy')
    print(f"Results saved to {results_dir}/detector_results_{file_basename}_{model_name}.csv")
    
    return file_basename, test_acc, test_auc, valid_acc, valid_auc, latency, energy

def inference_detector(model_path, inference_file_path, image_type='adversarial', batch_size=32):
    """
    Run inference on images using trained detector
    
    Args:
        model_path: Path to saved detector model (.pkl file)
        inference_file_path: Path to .adv file for inference
        image_type: 'adversarial' or 'original' - which images to extract
        batch_size: Batch size for feature extraction
    
    Returns:
        predictions, probabilities, accuracy, detailed_results
    """
    print(f"\n{'='*40}")
    print(f"Inference on {image_type} images")
    print(f"File: {os.path.basename(inference_file_path)}")
    print(f"{'='*40}")
    
    # Load trained model
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    svm_model = model_data['model']
    feature_extractor_name = model_data['feature_extractor']
    
    # Load inference data based on type
    if image_type == 'adversarial':
        images, labels = load_adv_data_inference(inference_file_path)
        expected_label = 1
    elif image_type == 'original':
        images, labels = load_original_data_inference(inference_file_path)
        expected_label = 0
    else:
        raise ValueError("image_type must be 'adversarial' or 'original'")
    
    # Create dataset with same transforms as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AdvImageDataset(images, labels, transform=transform)
    
    # Extract features using same feature extractor as training
    feature_extractor = FeatureExtractor(feature_extractor_name)
    features, true_labels = extract_features(dataset, feature_extractor, batch_size=batch_size)
    
    # Run inference
    start_time = time.time()
    predictions = svm_model.predict(features)
    probabilities = svm_model.predict_proba(features)[:, 1]  # Probability of being adversarial
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = np.mean(predictions == true_labels)
    
    # Detailed analysis
    n_total = len(predictions)
    n_correct = np.sum(predictions == true_labels)
    n_wrong = np.sum(predictions != true_labels)
    
    if image_type == 'adversarial':
        # For adversarial images, we want them detected as adversarial (1)
        n_detected = np.sum(predictions == 1)
        n_missed = np.sum(predictions == 0)
        detection_rate = n_detected / n_total
        print(f"  Total adversarial images: {n_total}")
        print(f"  Correctly detected as adversarial: {n_detected} ({detection_rate:.1%})")
        print(f"  Missed (predicted as benign): {n_missed} ({(1-detection_rate):.1%})")
    else:
        # For original images, we want them detected as benign (0)
        n_benign = np.sum(predictions == 0)
        n_false_positive = np.sum(predictions == 1)
        benign_rate = n_benign / n_total
        print(f"  Total original images: {n_total}")
        print(f"  Correctly identified as benign: {n_benign} ({benign_rate:.1%})")
        print(f"  False positives (predicted as adversarial): {n_false_positive} ({(1-benign_rate):.1%})")
    
    print(f"  Overall accuracy: {accuracy:.4f}")
    print(f"  Average adversarial probability: {np.mean(probabilities):.4f}")
    print(f"  Inference time: {inference_time:.4f}s ({inference_time/n_total*1000:.2f}ms per image)")
    
    # Create detailed results
    detailed_results = {
        'image_type': image_type,
        'n_total': n_total,
        'accuracy': accuracy,
        'avg_confidence': np.mean(probabilities),
        'inference_time': inference_time,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels
    }
    
    return predictions, probabilities, accuracy, detailed_results

def save_markdown_results(train_results, inference_results, model_info, file_info=None):
    """
    Save comprehensive results to markdown file
    
    Args:
        train_results: Dictionary with training results
        inference_results: List of inference results for different datasets
        model_info: Dictionary with model information
        file_info: Dictionary with file information (optional)
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_file = os.path.join(results_dir, f'results_{train_results["file_basename"]}_{model_info["model"]}_{timestamp}.md')
    
    with open(md_file, 'w') as f:
        f.write(f"# Adversarial Detector Results\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Information
        f.write(f"## Model Information\n\n")
        f.write(f"- **Feature Extractor:** {model_info['model']}\n")
        f.write(f"- **Training File:** {model_info['train_file']}\n")
        f.write(f"- **Validation Size:** {model_info['valid_size']}\n")
        f.write(f"- **Batch Size:** {model_info['batch_size']}\n\n")
        
        # File Information
        if file_info:
            f.write(f"## Test File Information\n\n")
            f.write(f"| File Type | Path | Exists | Total Images |\n")
            f.write(f"|-----------|------|--------|-------------|\n")
            for name, info in file_info.items():
                if info['provided']:
                    exists_str = "✓" if info['exists'] else "✗"
                    length_str = str(info['length']) if info['length'] is not None else "Error"
                    f.write(f"| {name.replace('_', ' ').title()} | `{info['path']}` | {exists_str} | {length_str} |\n")
                else:
                    f.write(f"| {name.replace('_', ' ').title()} | Not provided | - | - |\n")
            f.write(f"\n")
        
        # Training Results
        f.write(f"## Training Results\n\n")
        f.write(f"| Metric | Test | Validation |\n")
        f.write(f"|--------|------|------------|\n")
        f.write(f"| Accuracy | {train_results['test_acc']:.4f} | {train_results['valid_acc']:.4f} |\n")
        f.write(f"| AUC | {train_results['test_auc']:.4f} | {train_results['valid_auc']:.4f} |\n")
        f.write(f"| Latency | {train_results['latency']:.4f}s | - |\n\n")
        
        # Inference Results
        f.write(f"## Inference Results\n\n")
        
        for dataset_name, results in inference_results.items():
            f.write(f"### {dataset_name.replace('_', ' ').title()}\n\n")
            
            if 'adv' in results and 'original' in results:
                # Add sample counts
                adv_count = results['adv']['n_total']
                orig_count = results['original']['n_total']
                
                f.write(f"**Sample Counts:** {adv_count} adversarial, {orig_count} original\n\n")
                
                f.write(f"| Image Type | Accuracy | Avg Confidence | Inference Time |\n")
                f.write(f"|------------|----------|----------------|----------------|\n")
                f.write(f"| Adversarial | {results['adv']['accuracy']:.4f} | {results['adv']['avg_confidence']:.4f} | {results['adv']['inference_time']:.4f}s |\n")
                f.write(f"| Original | {results['original']['accuracy']:.4f} | {results['original']['avg_confidence']:.4f} | {results['original']['inference_time']:.4f}s |\n\n")
                
                # Additional metrics
                adv_acc = results['adv']['accuracy']
                orig_acc = results['original']['accuracy']
                f.write(f"**Detection Summary:**\n")
                f.write(f"- Adversarial Detection Rate: {adv_acc:.1%}\n")
                f.write(f"- Original Classification Rate: {orig_acc:.1%}\n")
                f.write(f"- False Positive Rate: {1-orig_acc:.1%}\n")
                f.write(f"- Overall Balance: {(adv_acc + orig_acc)/2:.4f}\n\n")
        
        # Summary
        f.write(f"## Summary\n\n")
        f.write(f"This detector was trained on {train_results['file_basename']} and evaluated on multiple test sets. ")
        f.write(f"The model achieved {train_results['test_acc']:.1%} test accuracy during training and showed ")
        f.write(f"varying performance across different attack types during inference.\n\n")
        
        f.write(f"**Key Findings:**\n")
        for dataset_name, results in inference_results.items():
            if 'adv' in results:
                adv_rate = results['adv']['accuracy']
                f.write(f"- {dataset_name}: {adv_rate:.1%} adversarial detection rate\n")
    
    print(f"Detailed results saved to: {md_file}")
    return md_file

def main():
    """
    Main function to train detector and run comprehensive inference
    """
    parser = argparse.ArgumentParser(description='Train adversarial detector and run comprehensive inference')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to .adv file for training')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'vit'],
                        help='Feature extractor model (default: resnet50)')
    parser.add_argument('--valid_size', type=int, default=100,
                        help='Number of samples for validation (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction (default: 32)')
    
    # Test files
    parser.add_argument('--test_file', type=str,
                        help='Path to main test .adv file (optional)')
    parser.add_argument('--test_jpeg_file', type=str,
                        help='Path to JPEG test .adv file (optional)')
    parser.add_argument('--test_spatial_file', type=str,
                        help='Path to spatial test .adv file (optional)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist!")
        return
    
    # Training phase
    print("Starting training phase...")
    print(f"Training file: {args.input_file}")
    print(f"Model: {args.model}")
    print(f"Validation size: {args.valid_size}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        file_basename, test_acc, test_auc, valid_acc, valid_auc, latency, energy = train_detector(
            args.input_file, 
            model_name=args.model, 
            valid_size=args.valid_size, 
            batch_size=args.batch_size
        )
        
        print(f"\nTraining Summary:")
        print(f"File: {file_basename}")
        print(f"Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
        print(f"Valid Accuracy: {valid_acc:.4f}, AUC: {valid_auc:.4f}")
        print(f"Latency: {latency:.4f}s, Energy: {energy}")
        
        # Prepare training results for markdown
        train_results = {
            'file_basename': file_basename,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'valid_acc': valid_acc,
            'valid_auc': valid_auc,
            'latency': latency,
            'energy': energy
        }
        
        model_info = {
            'model': args.model,
            'train_file': os.path.basename(args.input_file),
            'valid_size': args.valid_size,
            'batch_size': args.batch_size
        }
        
        # Inference phase
        test_files = {
            'test': args.test_file,
            'test_jpeg': args.test_jpeg_file,
            'test_spatial': args.test_spatial_file
        }
        
        # Debug: Show what files were provided
        print(f"\nProvided test files:")
        for name, path in test_files.items():
            if path is not None:
                exists = os.path.exists(path)
                print(f"  {name}: {path} (exists: {exists})")
                if exists:
                    try:
                        # Load file to get total number of images
                        data = torch.load(path, map_location='cpu', weights_only=False)
                        total_images = 0
                        for sample in data:
                            ori_img_list, adv_img_list = sample[0], sample[1]
                            ori_images = ori_img_list[0]  # Shape: [N, 3, 256, 256]
                            adv_images = adv_img_list[0]  # Shape: [N, 3, 256, 256]
                            total_images += ori_images.shape[0] + adv_images.shape[0]  # Count both original and adversarial
                        length = total_images
                        print(f"    -> Total images: {total_images} ({len(data)} pairs, {total_images//2} per type)")
                    except Exception as e:
                        length = f"Error: {e}"
                        print(f"    -> Error reading file: {e}")
                else:
                    print(f"    -> FILE NOT FOUND!")
            else:
                print(f"  {name}: None (not provided)")
        
        # Remove None values and create file info for markdown
        file_info = {}
        for name, path in test_files.items():
            if path is not None:
                exists = os.path.exists(path)
                length = None
                if exists:
                    try:
                        data = torch.load(path, map_location='cpu', weights_only=False)
                        total_images = 0
                        for sample in data:
                            ori_img_list, adv_img_list = sample[0], sample[1]
                            ori_images = ori_img_list[0]  # Shape: [N, 3, 256, 256]
                            adv_images = adv_img_list[0]  # Shape: [N, 3, 256, 256]
                            total_images += ori_images.shape[0] + adv_images.shape[0]  # Count both original and adversarial
                        length = total_images
                    except Exception as e:
                        length = f"Error: {e}"
                
                file_info[name] = {
                    'provided': True,
                    'path': path,
                    'exists': exists,
                    'length': length
                }
            else:
                file_info[name] = {
                    'provided': False,
                    'path': None,
                    'exists': False,
                    'length': None
                }
        
        test_files = {k: v for k, v in test_files.items() if v is not None}
        
        print(f"\nFiles to process: {list(test_files.keys())}")
        
        if test_files:
            print(f"\n{'='*60}")
            print("Starting comprehensive inference phase...")
            print(f"{'='*60}")
            
            # Construct model path from training
            model_path = os.path.join(model_dir, f'detector_{file_basename}_{args.model}.pkl')
            inference_results = {}
            
            for dataset_name, test_file in test_files.items():
                print(f"\nChecking {dataset_name}: {test_file}")
                
                if not os.path.exists(test_file):
                    print(f"Warning: Test file {test_file} does not exist! Skipping {dataset_name}.")
                    continue
                
                print(f"--- Processing {dataset_name} ---")
                
                try:
                    # Test on adversarial images
                    pred_adv, prob_adv, acc_adv, detail_adv = inference_detector(
                        model_path, test_file, 'adversarial', args.batch_size
                    )
                    
                    # Test on original images
                    pred_orig, prob_orig, acc_orig, detail_orig = inference_detector(
                        model_path, test_file, 'original', args.batch_size
                    )
                    
                    # Store results
                    inference_results[dataset_name] = {
                        'adv': detail_adv,
                        'original': detail_orig
                    }
                    
                    # Save individual CSV results
                    inference_basename = get_file_basename(test_file)
                    model_basename = get_file_basename(model_path)
                    
                    # Save adversarial results
                    adv_file = os.path.join(results_dir, f'inference_{model_basename}_{inference_basename}_adv.csv')
                    adv_results = np.column_stack([
                        range(len(pred_adv)), pred_adv, prob_adv, detail_adv['true_labels']
                    ])
                    np.savetxt(adv_file, adv_results, delimiter=',', fmt='%d,%d,%.6f,%d',
                               header='image_index,predicted_label,adversarial_probability,true_label')
                    
                    # Save original results
                    orig_file = os.path.join(results_dir, f'inference_{model_basename}_{inference_basename}_orig.csv')
                    orig_results = np.column_stack([
                        range(len(pred_orig)), pred_orig, prob_orig, detail_orig['true_labels']
                    ])
                    np.savetxt(orig_file, orig_results, delimiter=',', fmt='%d,%d,%.6f,%d',
                               header='image_index,predicted_label,adversarial_probability,true_label')
                    
                except Exception as e:
                    print(f"Error during inference on {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Generate final summary
            if inference_results:
                print(f"\n{'='*60}")
                print("COMPREHENSIVE RESULTS SUMMARY")
                print(f"{'='*60}")
                print(f"Training - Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
                print(f"Training - Valid Accuracy: {valid_acc:.4f}, AUC: {valid_auc:.4f}")
                
                for dataset_name, results in inference_results.items():
                    adv_acc = results['adv']['accuracy']
                    orig_acc = results['original']['accuracy']
                    adv_conf = results['adv']['avg_confidence']
                    orig_conf = results['original']['avg_confidence']
                    
                    print(f"\n{dataset_name.upper()}:")
                    print(f"  Adversarial Detection: {adv_acc:.4f} (confidence: {adv_conf:.4f})")
                    print(f"  Original Classification: {orig_acc:.4f} (confidence: {orig_conf:.4f})")
                    print(f"  Balance Score: {(adv_acc + orig_acc)/2:.4f}")
                
                # Save markdown results
                md_file = save_markdown_results(train_results, inference_results, model_info, file_info)
                
        else:
            print(f"\nNo test files provided. Training completed.")
            print(f"To run inference later, provide --test_file, --test_jpeg_file, or --test_spatial_file arguments.")
        
    except Exception as e:
        print(f"Error training detector: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()