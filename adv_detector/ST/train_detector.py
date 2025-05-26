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
import glob


model_dir = 'model'
results_dir = 'results'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDataset(Dataset):
    """Dataset class for loading images from file paths"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
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

def get_image_files(folder_path, extensions=['jpg', 'jpeg', 'png', 'bmp', 'tiff']):
    """Get all image files from a folder"""
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, f"*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase
        pattern = os.path.join(folder_path, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    
    # Sort to ensure consistent ordering
    image_files.sort()
    return image_files

def get_filename_without_ext(filepath):
    """Get filename without extension"""
    return os.path.splitext(os.path.basename(filepath))[0]

def load_paired_data(clean_folder, adv_folder, max_pairs=None):
    """
    Load paired clean and adversarial images
    
    Args:
        clean_folder: Path to folder containing clean images
        adv_folder: Path to folder containing adversarial images
        max_pairs: Maximum number of pairs to use (uses minimum of available if None)
    
    Returns:
        image_paths: List of image file paths
        labels: List of labels (0=original, 1=adversarial)
        pairs: List of (clean_path, adv_path) tuples for matched pairs
    """
    print(f"Loading clean images from: {clean_folder}")
    print(f"Loading adversarial images from: {adv_folder}")
    
    # Get all image files (sorted for consistent ordering)
    clean_files = get_image_files(clean_folder)
    adv_files = get_image_files(adv_folder)
    
    print(f"Found {len(clean_files)} clean images")
    print(f"Found {len(adv_files)} adversarial images")
    
    # Determine how many pairs to use
    min_available = min(len(clean_files), len(adv_files))
    
    if max_pairs is not None:
        n_pairs = min(max_pairs, min_available)
        print(f"Using first {n_pairs} pairs (max_pairs={max_pairs}, available={min_available})")
    else:
        n_pairs = min_available
        print(f"Using {n_pairs} pairs (limited by smaller dataset)")
    
    if n_pairs == 0:
        raise ValueError("No images found in one or both folders!")
    
    # Take the first n_pairs from each folder (assumes they're paired by position)
    clean_files_to_use = clean_files[:n_pairs]
    adv_files_to_use = adv_files[:n_pairs]
    
    print(f"Selected images:")
    print(f"  Clean: {len(clean_files_to_use)} images")
    print(f"  Adversarial: {len(adv_files_to_use)} images")
    print(f"  First clean: {os.path.basename(clean_files_to_use[0])}")
    print(f"  First adversarial: {os.path.basename(adv_files_to_use[0])}")
    print(f"  Last clean: {os.path.basename(clean_files_to_use[-1])}")
    print(f"  Last adversarial: {os.path.basename(adv_files_to_use[-1])}")
    
    # Create paired data by position
    all_image_paths = []
    all_labels = []
    pairs = []
    
    for i in range(n_pairs):
        clean_path = clean_files_to_use[i]
        adv_path = adv_files_to_use[i]
        
        all_image_paths.append(clean_path)
        all_labels.append(0)  # Clean = 0
        
        all_image_paths.append(adv_path)
        all_labels.append(1)  # Adversarial = 1
        
        pairs.append((clean_path, adv_path))
    
    print(f"\nCreated dataset with {len(all_image_paths)} images ({len(pairs)} pairs)")
    print(f"Clean images: {all_labels.count(0)}")
    print(f"Adversarial images: {all_labels.count(1)}")
    
    return all_image_paths, all_labels, pairs

def load_inference_data(folder_path, image_type='adversarial'):
    """
    Load images for inference
    
    Args:
        folder_path: Path to folder containing images
        image_type: 'adversarial' or 'original' - for labeling purposes
    
    Returns:
        image_paths: List of image file paths
        labels: List of labels (all same based on image_type)
    """
    print(f"Loading {image_type} images from: {folder_path}")
    
    image_files = get_image_files(folder_path)
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path}")
    
    # Create labels based on image type
    label = 1 if image_type == 'adversarial' else 0
    labels = [label] * len(image_files)
    
    return image_files, labels

def extract_features(dataset, feature_extractor, batch_size=32):
    """Extract features from images using the feature extractor"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
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

def verify_training_labels(image_paths, labels, pairs, n_samples_to_check=10):
    """
    Verify that training labels are correctly assigned
    
    Args:
        image_paths: List of image file paths
        labels: List of labels (0=clean, 1=adversarial)
        pairs: List of (clean_path, adv_path) tuples
        n_samples_to_check: Number of samples to manually verify
    """
    print(f"\n{'='*50}")
    print("TRAINING LABEL VERIFICATION")
    print(f"{'='*50}")
    
    # Overall statistics
    n_total = len(labels)
    n_clean = labels.count(0)
    n_adv = labels.count(1)
    
    print(f"Total samples: {n_total}")
    print(f"Clean samples (label=0): {n_clean}")
    print(f"Adversarial samples (label=1): {n_adv}")
    print(f"Expected ratio: 50/50 -> Actual: {n_clean/n_total:.1%}/{n_adv/n_total:.1%}")
    
    # Check alternating pattern
    print(f"\nChecking alternating pattern (should be [0,1,0,1,...]):")
    pattern_correct = True
    for i in range(min(20, len(labels))):
        expected = i % 2  # 0,1,0,1,...
        actual = labels[i]
        status = "✓" if expected == actual else "✗"
        print(f"  Index {i}: Expected {expected}, Got {actual} {status}")
        if expected != actual:
            pattern_correct = False
    
    if pattern_correct:
        print("  Pattern verification: PASSED ✓")
    else:
        print("  Pattern verification: FAILED ✗")
    
    # Sample verification - check specific pairs
    print(f"\nSample pair verification:")
    indices_to_check = np.linspace(0, min(len(pairs)-1, n_samples_to_check-1), n_samples_to_check, dtype=int)
    
    for idx in indices_to_check:
        pair_idx = idx
        clean_path, adv_path = pairs[pair_idx]
        
        # Find corresponding sample indices
        clean_sample_idx = pair_idx * 2
        adv_sample_idx = pair_idx * 2 + 1
        
        clean_path_in_list = image_paths[clean_sample_idx]
        adv_path_in_list = image_paths[adv_sample_idx]
        clean_label = labels[clean_sample_idx]
        adv_label = labels[adv_sample_idx]
        
        clean_match = clean_path == clean_path_in_list
        adv_match = adv_path == adv_path_in_list
        clean_label_correct = clean_label == 0
        adv_label_correct = adv_label == 1
        
        print(f"  Pair {pair_idx}:")
        print(f"    Clean: {os.path.basename(clean_path)} -> label={clean_label} {'✓' if clean_label_correct else '✗'}")
        print(f"    Adv:   {os.path.basename(adv_path)} -> label={adv_label} {'✓' if adv_label_correct else '✗'}")
        print(f"    Path match: clean={'✓' if clean_match else '✗'}, adv={'✓' if adv_match else '✗'}")
    
    # Summary
    all_checks_passed = (
        n_clean == n_adv and  # Equal numbers
        pattern_correct and   # Alternating pattern
        all(labels[i*2] == 0 and labels[i*2+1] == 1 for i in range(min(10, len(pairs))))  # Sample pairs correct
    )
    
    print(f"\n{'='*50}")
    if all_checks_passed:
        print("TRAINING LABEL VERIFICATION: PASSED ✓")
        print("All labels are correctly assigned!")
    else:
        print("TRAINING LABEL VERIFICATION: FAILED ✗")
        print("There may be issues with label assignment!")
    print(f"{'='*50}")
    
    return all_checks_passed
    """Extract basename from folder path for naming saved files"""
    return os.path.basename(os.path.normpath(folder_path))

def train_detector(clean_folder, adv_folder, model_name='resnet50', train_ratio=0.7, valid_size=100, batch_size=32, random_seed=42, max_pairs=None):
    """
    Train a detector for a specific attack type
    
    Args:
        clean_folder: Path to folder with clean images
        adv_folder: Path to folder with adversarial images
        model_name: Feature extractor model ('resnet50' or 'vit')
        train_ratio: Ratio of data to use for training (rest goes to test, except validation)
        valid_size: Number of samples for validation
        batch_size: Batch size for feature extraction
        random_seed: Random seed for reproducibility
        max_pairs: Maximum number of image pairs to use (None = use all available)
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    folder_basename = f"{get_folder_basename(clean_folder)}_vs_{get_folder_basename(adv_folder)}"
    
    print(f"\n{'='*60}")
    print(f"Training detector for {folder_basename} using {model_name}")
    print(f"{'='*60}")
    
def get_folder_basename(folder_path):
    """Extract basename from folder path for naming saved files"""
    return os.path.basename(os.path.normpath(folder_path))

def train_detector(clean_folder, adv_folder, model_name='resnet50', train_ratio=0.7, valid_size=100, batch_size=32, random_seed=42, max_pairs=None):
    """
    Train a detector for a specific attack type
    
    Args:
        clean_folder: Path to folder with clean images
        adv_folder: Path to folder with adversarial images
        model_name: Feature extractor model ('resnet50' or 'vit')
        train_ratio: Ratio of data to use for training (rest goes to test, except validation)
        valid_size: Number of samples for validation
        batch_size: Batch size for feature extraction
        random_seed: Random seed for reproducibility
        max_pairs: Maximum number of image pairs to use (None = use all available)
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    folder_basename = f"{get_folder_basename(clean_folder)}_vs_{get_folder_basename(adv_folder)}"
    
    print(f"\n{'='*60}")
    print(f"Training detector for {folder_basename} using {model_name}")
    print(f"{'='*60}")
    
    # Load paired data with optional limit
    image_paths, labels, pairs = load_paired_data(clean_folder, adv_folder, max_pairs=max_pairs)
    
    # Verify training labels are correct
    verify_training_labels(image_paths, labels, pairs)
    
    # Create dataset with transforms for feature extraction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(image_paths, labels, transform=transform)
    
    # Extract features
    feature_extractor = FeatureExtractor(model_name)
    features, labels = extract_features(dataset, feature_extractor, batch_size=batch_size)
    
    # Split data: validation + train/test split
    # We need to ensure pairs stay together, so we split at pair level
    n_pairs = len(pairs)
    pair_indices = list(range(n_pairs))
    random.shuffle(pair_indices)
    
    # First take out validation pairs
    valid_pairs = valid_size // 2  # Since each pair gives 2 samples
    valid_pair_indices = pair_indices[:valid_pairs]
    remaining_pair_indices = pair_indices[valid_pairs:]
    
    # Then split remaining pairs into train/test
    n_remaining_pairs = len(remaining_pair_indices)
    test_pairs = int(n_remaining_pairs * (1 - train_ratio))
    test_pair_indices = remaining_pair_indices[:test_pairs]
    train_pair_indices = remaining_pair_indices[test_pairs:]
    
    # Convert pair indices to sample indices (each pair has 2 samples: clean and adv)
    def pair_indices_to_sample_indices(pair_indices):
        sample_indices = []
        for pair_idx in pair_indices:
            # Each pair contributes 2 consecutive samples (clean, then adv)
            sample_indices.extend([pair_idx * 2, pair_idx * 2 + 1])
        return sample_indices
    
    train_indices = pair_indices_to_sample_indices(train_pair_indices)
    test_indices = pair_indices_to_sample_indices(test_pair_indices)
    valid_indices = pair_indices_to_sample_indices(valid_pair_indices)
    
    # Create splits
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    valid_features = features[valid_indices]
    valid_labels = labels[valid_indices]
    
    print(f"Data split:")
    print(f"  Training: {len(train_features)} samples ({len(train_pair_indices)} pairs)")
    print(f"  Testing: {len(test_features)} samples ({len(test_pair_indices)} pairs)")
    print(f"  Validation: {len(valid_features)} samples ({len(valid_pair_indices)} pairs)")
    
    # Train SVM
    print("Training SVM classifier...")
    svm_model = SVC(probability=True, kernel='rbf', random_state=random_seed, verbose=True)
    
    # Create a simple progress bar for SVM training
    print("  Starting SVM optimization...")
    start_time = time.time()
    svm_model.fit(train_features, train_labels)
    end_time = time.time()
    print(f"  SVM training completed in {end_time - start_time:.2f} seconds")
    
    # Save model and data
    save_name = os.path.join(model_dir, f'detector_{folder_basename}_{model_name}.pkl')
    torch.save({
        'model': svm_model,
        'train_data': (train_features, train_labels),
        'test_data': (test_features, test_labels),
        'valid_data': (valid_features, valid_labels),
        'feature_extractor': model_name,
        'clean_folder': clean_folder,
        'adv_folder': adv_folder,
        'pairs': pairs
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
    
    print(f"\nResults for {folder_basename}:")
    print(f"  Class distribution - Test: {test_class_dist}, Valid: {valid_class_dist}")
    print(f"  Default threshold (0.5):")
    print(f"    Test  - Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
    print(f"    Valid - Accuracy: {valid_acc:.4f}, AUC: {valid_auc:.4f}")
    print(f"  Optimal threshold ({best_threshold:.3f}):")
    print(f"    Test  - Accuracy: {test_acc_optimal:.4f}")
    print(f"    Valid - Accuracy: {best_acc:.4f}")
    print(f"  Latency: {latency:.4f}s, Energy: {energy}")
    
    # Save results
    results = [[folder_basename, test_acc, test_auc, valid_acc, valid_auc, latency, energy]]
    os.makedirs(results_dir, exist_ok=True)
    np.savetxt(f'{results_dir}/detector_results_{folder_basename}_{model_name}.csv', 
               results, delimiter=',', fmt='%s',
               header='attack_id,test_acc,test_auc,valid_acc,valid_auc,latency,energy')
    print(f"Results saved to {results_dir}/detector_results_{folder_basename}_{model_name}.csv")
    
    return folder_basename, test_acc, test_auc, valid_acc, valid_auc, latency, energy

def inference_detector(model_path, test_folder, image_type='adversarial', batch_size=32, expected_label=None):
    """
    Run inference on images using trained detector
    
    Args:
        model_path: Path to saved detector model (.pkl file)
        test_folder: Path to folder containing test images
        image_type: 'adversarial' or 'original' - which type of images we're testing
        batch_size: Batch size for feature extraction
        expected_label: Expected label (0=clean, 1=adversarial). If None, inferred from image_type
    
    Returns:
        predictions, probabilities, accuracy, detailed_results
    """
    print(f"\n{'='*40}")
    print(f"Inference on {image_type} images")
    print(f"Folder: {test_folder}")
    print(f"{'='*40}")
    
    # Load trained model
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    svm_model = model_data['model']
    feature_extractor_name = model_data['feature_extractor']
    
    # Load inference data
    image_paths, labels = load_inference_data(test_folder, image_type)
    
    # Override labels if expected_label is provided
    if expected_label is not None:
        print(f"Overriding labels: Using expected_label={expected_label} ({'clean' if expected_label == 0 else 'adversarial'})")
        labels = [expected_label] * len(image_paths)
        # Update image_type for display purposes
        actual_image_type = 'original' if expected_label == 0 else 'adversarial'
        if actual_image_type != image_type:
            print(f"Note: image_type='{image_type}' but expected_label={expected_label} ('{actual_image_type}')")
    
    # Create dataset with same transforms as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(image_paths, labels, transform=transform)
    
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
    
    # Determine what we're expecting based on true labels
    expected_class = true_labels[0]  # Should be all the same
    
    if expected_class == 1:  # Adversarial images
        n_detected = np.sum(predictions == 1)
        n_missed = np.sum(predictions == 0)
        detection_rate = n_detected / n_total
        print(f"  Total adversarial images: {n_total}")
        print(f"  Correctly detected as adversarial: {n_detected} ({detection_rate:.1%})")
        print(f"  Missed (predicted as benign): {n_missed} ({(1-detection_rate):.1%})")
    else:  # Original images
        n_benign = np.sum(predictions == 0)
        n_false_positive = np.sum(predictions == 1)
        benign_rate = n_benign / n_total
        print(f"  Total original images: {n_total}")
        print(f"  Correctly identified as benign: {n_benign} ({benign_rate:.1%})")
        print(f"  False positives (predicted as adversarial): {n_false_positive} ({(1-benign_rate):.1%})")
    
    print(f"  Overall accuracy: {accuracy:.4f}")
    print(f"  Average adversarial probability: {np.mean(probabilities):.4f}")
    print(f"  Inference time: {inference_time:.4f}s ({inference_time/n_total*1000:.2f}ms per image)")
    
    # Verify label consistency
    unique_labels = np.unique(true_labels)
    if len(unique_labels) > 1:
        print(f"  WARNING: Mixed labels detected! Labels: {unique_labels}")
    else:
        print(f"  Label consistency: All labels are {unique_labels[0]} ({'clean' if unique_labels[0] == 0 else 'adversarial'}) ✓")
    
    # Create detailed results
    detailed_results = {
        'image_type': image_type,
        'expected_label': expected_class,
        'n_total': n_total,
        'accuracy': accuracy,
        'avg_confidence': np.mean(probabilities),
        'inference_time': inference_time,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels,
        'image_paths': image_paths
    }
    
    return predictions, probabilities, accuracy, detailed_results

def save_markdown_results(train_results, inference_results, model_info, folder_info=None):
    """
    Save comprehensive results to markdown file matching console output format
    
    Args:
        train_results: Dictionary with training results
        inference_results: Dictionary of inference results for different datasets
        model_info: Dictionary with model information
        folder_info: Dictionary with folder information (optional)
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_file = os.path.join(results_dir, f'results_{train_results["folder_basename"]}_{model_info["model"]}_{timestamp}.md')
    
    with open(md_file, 'w') as f:
        f.write(f"# Adversarial Detector Results\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Information
        f.write(f"## Model Information\n\n")
        f.write(f"- **Feature Extractor:** {model_info['model']}\n")
        f.write(f"- **Clean Images Folder:** {model_info['clean_folder']}\n")
        f.write(f"- **Adversarial Images Folder:** {model_info['adv_folder']}\n")
        f.write(f"- **Train Ratio:** {model_info['train_ratio']}\n")
        f.write(f"- **Validation Size:** {model_info['valid_size']}\n")
        f.write(f"- **Batch Size:** {model_info['batch_size']}\n\n")
        
        # External Folder Information (if any provided)
        if folder_info:
            has_external = any(info['provided'] and info['exists'] for info in folder_info.values())
            if has_external:
                f.write(f"## External Test Folders\n\n")
                f.write(f"| Folder Type | Path | Total Images |\n")
                f.write(f"|-------------|------|-------------|\n")
                for name, info in folder_info.items():
                    if info['provided'] and info['exists']:
                        length_str = str(info['length']) if info['length'] is not None else "Error"
                        f.write(f"| {name.replace('_', ' ').title()} | `{os.path.basename(info['path'])}` | {length_str} |\n")
                f.write(f"\n")
        
        # Training Results
        f.write(f"## Training Results\n\n")
        f.write(f"- **Test Accuracy:** {train_results['test_acc']:.4f}\n")
        f.write(f"- **Test AUC:** {train_results['test_auc']:.4f}\n")
        f.write(f"- **Validation Accuracy:** {train_results['valid_acc']:.4f}\n")
        f.write(f"- **Validation AUC:** {train_results['valid_auc']:.4f}\n\n")
        
        # Internal Test Split Results
        if 'internal_test' in inference_results:
            internal = inference_results['internal_test']
            f.write(f"## Internal Test Split Results\n\n")
            
            if 'original' in internal and 'adv' in internal:
                clean_acc = internal['original']['accuracy']
                clean_samples = internal['original']['n_total']
                adv_acc = internal['adv']['accuracy']
                adv_samples = internal['adv']['n_total']
                overall_acc = train_results['test_acc']
                
                f.write(f"- **Clean Images:** {clean_acc:.4f} accuracy ({clean_samples} samples)\n")
                f.write(f"- **Adversarial Images:** {adv_acc:.4f} accuracy ({adv_samples} samples)\n")
                f.write(f"- **Overall:** {overall_acc:.4f} accuracy\n\n")
        
        # External Dataset Results
        external_results = {k: v for k, v in inference_results.items() if k != 'internal_test'}
        if external_results:
            f.write(f"## External Dataset Results\n\n")
            
            for dataset_name, results in external_results.items():
                for img_type, detail in results.items():
                    accuracy = detail['accuracy']
                    confidence = detail['avg_confidence']
                    n_samples = detail['n_total']
                    clean_name = dataset_name.replace('_', ' ').title()
                    type_name = img_type.replace('_', ' ').title()
                    
                    f.write(f"- **{clean_name} ({type_name}):** {accuracy:.4f} accuracy, {confidence:.4f} avg confidence ({n_samples} samples)\n")
            f.write(f"\n")
        
        # Detailed Performance Table
        f.write(f"## Detailed Performance Summary\n\n")
        f.write(f"| Dataset | Image Type | Accuracy | Avg Confidence | Sample Count |\n")
        f.write(f"|---------|------------|----------|----------------|-------------|\n")
        
        # Add internal test results
        if 'internal_test' in inference_results:
            internal = inference_results['internal_test']
            if 'original' in internal:
                clean = internal['original']
                f.write(f"| Internal Test | Clean | {clean['accuracy']:.4f} | {clean['avg_confidence']:.4f} | {clean['n_total']} |\n")
            if 'adv' in internal:
                adv = internal['adv']
                f.write(f"| Internal Test | Adversarial | {adv['accuracy']:.4f} | {adv['avg_confidence']:.4f} | {adv['n_total']} |\n")
        
        # Add external results
        for dataset_name, results in external_results.items():
            clean_dataset_name = dataset_name.replace('_', ' ').title()
            for img_type, detail in results.items():
                clean_img_type = img_type.replace('_', ' ').title()
                f.write(f"| {clean_dataset_name} | {clean_img_type} | {detail['accuracy']:.4f} | {detail['avg_confidence']:.4f} | {detail['n_total']} |\n")
        
        f.write(f"\n")
        
        # Key Insights
        f.write(f"## Key Insights\n\n")
        
        # Training performance
        f.write(f"### Training Performance\n")
        f.write(f"The detector achieved **{train_results['test_acc']:.1%}** accuracy on the internal test split, ")
        f.write(f"with an AUC of **{train_results['test_auc']:.3f}**, indicating ")
        if train_results['test_auc'] >= 0.9:
            f.write(f"excellent discriminative performance.\n\n")
        elif train_results['test_auc'] >= 0.8:
            f.write(f"good discriminative performance.\n\n")
        else:
            f.write(f"moderate discriminative performance.\n\n")
        
        # Internal test performance
        if 'internal_test' in inference_results:
            internal = inference_results['internal_test']
            if 'original' in internal and 'adv' in internal:
                clean_acc = internal['original']['accuracy']
                adv_acc = internal['adv']['accuracy']
                
                f.write(f"### Detection Performance\n")
                f.write(f"- **Clean image classification:** {clean_acc:.1%} (lower false positive rate is better)\n")
                f.write(f"- **Adversarial detection:** {adv_acc:.1%} (higher detection rate is better)\n")
                
                if clean_acc >= 0.9 and adv_acc >= 0.9:
                    f.write(f"- **Overall assessment:** Excellent performance on both clean and adversarial images\n\n")
                elif clean_acc >= 0.8 and adv_acc >= 0.8:
                    f.write(f"- **Overall assessment:** Good balanced performance\n\n")
                else:
                    f.write(f"- **Overall assessment:** Performance may need improvement\n\n")
        
        # External dataset insights
        if external_results:
            f.write(f"### Robustness Analysis\n")
            for dataset_name, results in external_results.items():
                for img_type, detail in results.items():
                    accuracy = detail['accuracy']
                    clean_name = dataset_name.replace('_', ' ')
                    
                    if 'jpeg' in dataset_name:
                        f.write(f"- **JPEG robustness:** {accuracy:.1%} detection rate - ")
                        if accuracy >= 0.8:
                            f.write(f"detector is robust to JPEG compression\n")
                        elif accuracy >= 0.6:
                            f.write(f"detector shows moderate robustness to JPEG compression\n")
                        else:
                            f.write(f"detector struggles with JPEG-compressed adversarial examples\n")
                    
                    elif 'spatial' in dataset_name:
                        f.write(f"- **Spatial robustness:** {accuracy:.1%} detection rate - ")
                        if accuracy >= 0.8:
                            f.write(f"detector is robust to spatial transformations\n")
                        elif accuracy >= 0.6:
                            f.write(f"detector shows moderate robustness to spatial transformations\n")
                        else:
                            f.write(f"detector struggles with spatially-transformed adversarial examples\n")
                    
                    else:
                        f.write(f"- **{clean_name}:** {accuracy:.1%} detection rate\n")
            f.write(f"\n")
        
        # Recommendations
        f.write(f"## Recommendations\n\n")
        
        if 'internal_test' in inference_results:
            internal = inference_results['internal_test']
            if 'original' in internal and 'adv' in internal:
                clean_acc = internal['original']['accuracy']
                adv_acc = internal['adv']['accuracy']
                
                if clean_acc < 0.9:
                    f.write(f"- Consider improving clean image classification to reduce false positives\n")
                if adv_acc < 0.9:
                    f.write(f"- Consider improving adversarial detection capabilities\n")
                if abs(clean_acc - adv_acc) > 0.1:
                    f.write(f"- Consider balancing the dataset or adjusting the classification threshold\n")
        
        # Robustness recommendations
        low_robustness = []
        if external_results:
            for dataset_name, results in external_results.items():
                for img_type, detail in results.items():
                    if detail['accuracy'] < 0.7:
                        low_robustness.append(dataset_name.replace('_', ' '))
        
        if low_robustness:
            f.write(f"- Consider data augmentation with {', '.join(low_robustness)} to improve robustness\n")
        
        if not external_results:
            f.write(f"- Test robustness against JPEG compression and spatial transformations\n")
        
        f.write(f"- Consider ensemble methods or more advanced feature extractors for improved performance\n")
    
    print(f"Detailed results saved to: {md_file}")
    return md_file

def main():
    """
    Main function to train detector and run comprehensive inference
    """
    parser = argparse.ArgumentParser(description='Train adversarial detector on image folders and run comprehensive inference')
    
    # Training arguments
    parser.add_argument('--clean_folder', type=str, required=True,
                        help='Path to folder containing clean images')
    parser.add_argument('--adv_folder', type=str, required=True,
                        help='Path to folder containing adversarial images')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'vit'],
                        help='Feature extractor model (default: resnet50)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--valid_size', type=int, default=100,
                        help='Number of samples for validation (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction (default: 32)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Maximum number of image pairs to use (default: None = use all available)')
    
    # Inference folders (optional - for testing on different datasets)
    parser.add_argument('--inference_adv_folder', type=str,
                        help='Path to adversarial images for inference (optional)')
    parser.add_argument('--inference_clean_folder', type=str,
                        help='Path to clean images for inference (optional)')
    parser.add_argument('--inference_jpeg_folder', type=str,
                        help='Path to JPEG images for inference (optional)')
    parser.add_argument('--inference_spatial_folder', type=str,
                        help='Path to spatial images for inference (optional)')
    
    # Labels for inference folders (0=clean, 1=adversarial)
    parser.add_argument('--inference_jpeg_label', type=int, choices=[0, 1], default=1,
                        help='Label for JPEG images: 0=clean, 1=adversarial (default: 1)')
    parser.add_argument('--inference_spatial_label', type=int, choices=[0, 1], default=1,
                        help='Label for spatial images: 0=clean, 1=adversarial (default: 1)')
    
    args = parser.parse_args()
    
    # Check if input folders exist
    if not os.path.exists(args.clean_folder):
        print(f"Error: Clean folder {args.clean_folder} does not exist!")
        return
    
    if not os.path.exists(args.adv_folder):
        print(f"Error: Adversarial folder {args.adv_folder} does not exist!")
        return
    
    # Training phase
    print("Starting training phase...")
    print(f"Clean folder: {args.clean_folder}")
    print(f"Adversarial folder: {args.adv_folder}")
    print(f"Model: {args.model}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Validation size: {args.valid_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.random_seed}")
    
    try:
        folder_basename, test_acc, test_auc, valid_acc, valid_auc, latency, energy = train_detector(
            args.clean_folder,
            args.adv_folder,
            model_name=args.model, 
            train_ratio=args.train_ratio,
            valid_size=args.valid_size, 
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            max_pairs=args.max_pairs
        )
        
        print(f"\nTraining Summary:")
        print(f"Dataset: {folder_basename}")
        print(f"Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
        print(f"Valid Accuracy: {valid_acc:.4f}, AUC: {valid_auc:.4f}")
        print(f"Latency: {latency:.4f}s, Energy: {energy}")
        
        # Prepare training results for markdown
        train_results = {
            'folder_basename': folder_basename,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'valid_acc': valid_acc,
            'valid_auc': valid_auc,
            'latency': latency,
            'energy': energy
        }
        
        model_info = {
            'model': args.model,
            'clean_folder': os.path.basename(args.clean_folder),
            'adv_folder': os.path.basename(args.adv_folder),
            'train_ratio': args.train_ratio,
            'valid_size': args.valid_size,
            'batch_size': args.batch_size
        }
        
        # Inference phase - Test on the test split from training data
        print(f"\n{'='*60}")
        print("INFERENCE PHASE")
        print(f"{'='*60}")
        
        # Load the trained model
        model_path = os.path.join(model_dir, f'detector_{folder_basename}_{args.model}.pkl')
        model_data = torch.load(model_path, map_location=device, weights_only=False)
        svm_model = model_data['model']
        
        # Get test data (this was already split during training)
        test_features, test_labels = model_data['test_data']
        
        print(f"\n1. Testing on internal test split:")
        print(f"   Total test samples: {len(test_features)}")
        
        # Run inference on test split
        start_time = time.time()
        test_predictions = svm_model.predict(test_features)
        test_probabilities = svm_model.predict_proba(test_features)[:, 1]
        inference_time = time.time() - start_time
        
        # Separate clean and adversarial results from test split
        clean_indices = np.where(test_labels == 0)[0]
        adv_indices = np.where(test_labels == 1)[0]
        
        # Clean test results
        clean_predictions = test_predictions[clean_indices]
        clean_probabilities = test_probabilities[clean_indices]
        clean_labels = test_labels[clean_indices]
        clean_accuracy = np.mean(clean_predictions == clean_labels)
        
        # Adversarial test results  
        adv_predictions = test_predictions[adv_indices]
        adv_probabilities = test_probabilities[adv_indices]
        adv_labels = test_labels[adv_indices]
        adv_accuracy = np.mean(adv_predictions == adv_labels)
        
        print(f"   Clean images in test: {len(clean_indices)} - Accuracy: {clean_accuracy:.4f}")
        print(f"   Adversarial images in test: {len(adv_indices)} - Accuracy: {adv_accuracy:.4f}")
        print(f"   Overall test accuracy: {test_acc:.4f}")
        
        # Store internal test results
        inference_results = {
            'internal_test': {
                'adv': {
                    'image_type': 'adversarial',
                    'n_total': len(adv_indices),
                    'accuracy': adv_accuracy,
                    'avg_confidence': np.mean(adv_probabilities),
                    'inference_time': inference_time * len(adv_indices) / len(test_features),
                    'predictions': adv_predictions,
                    'probabilities': adv_probabilities,
                    'true_labels': adv_labels
                },
                'original': {
                    'image_type': 'original',
                    'n_total': len(clean_indices),
                    'accuracy': clean_accuracy,
                    'avg_confidence': np.mean(clean_probabilities),
                    'inference_time': inference_time * len(clean_indices) / len(test_features),
                    'predictions': clean_predictions,
                    'probabilities': clean_probabilities,
                    'true_labels': clean_labels
                }
            }
        }
        
        # Save internal test results to CSV
        internal_csv_file = os.path.join(results_dir, f'internal_test_{folder_basename}_{args.model}.csv')
        internal_results = []
        for i, (pred, prob, true_label) in enumerate(zip(test_predictions, test_probabilities, test_labels)):
            img_type = 'adversarial' if true_label == 1 else 'clean'
            internal_results.append([i, img_type, pred, prob, true_label])
        
        import csv
        with open(internal_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_index', 'image_type', 'predicted_label', 'adversarial_probability', 'true_label'])
            writer.writerows(internal_results)
        print(f"   Internal test results saved to: {internal_csv_file}")
        
        # Optional external inference folders (JPEG, Spatial, etc.)
        external_folders = {
            'jpeg': args.inference_jpeg_folder,
            'spatial': args.inference_spatial_folder,
            'external_adv': args.inference_adv_folder,
            'external_clean': args.inference_clean_folder
        }
        
        # Define expected labels for each folder type
        external_labels = {
            'jpeg': args.inference_jpeg_label,
            'spatial': args.inference_spatial_label,
            'external_adv': 1,  # Always adversarial
            'external_clean': 0  # Always clean
        }
        
        # Debug: Show what external folders were provided
        print(f"\n2. External inference folders:")
        has_external = False
        for name, path in external_folders.items():
            if path is not None:
                exists = os.path.exists(path)
                print(f"   {name}: {path} (exists: {exists})")
                if exists:
                    try:
                        image_files = get_image_files(path)
                        count = len(image_files)
                        print(f"      -> Total images: {count}")
                        has_external = True
                    except Exception as e:
                        print(f"      -> Error reading folder: {e}")
                else:
                    print(f"      -> FOLDER NOT FOUND!")
            else:
                print(f"   {name}: Not provided")
        
        # Create folder info for markdown
        folder_info = {}
        for name, path in external_folders.items():
            if path is not None:
                exists = os.path.exists(path)
                length = None
                if exists:
                    try:
                        image_files = get_image_files(path)
                        length = len(image_files)
                    except Exception as e:
                        length = f"Error: {e}"
                
                folder_info[name] = {
                    'provided': True,
                    'path': path,
                    'exists': exists,
                    'length': length
                }
            else:
                folder_info[name] = {
                    'provided': False,
                    'path': None,
                    'exists': False,
                    'length': None
                }
        
        # Process external folders if provided
        if has_external:
            print(f"\n3. Processing external datasets...")
            
            for dataset_name, folder_path in external_folders.items():
                if folder_path is None or not os.path.exists(folder_path):
                    continue
                
                print(f"\n--- Processing {dataset_name} ---")
                
                try:
                    # Get expected label for this dataset
                    expected_label = external_labels[dataset_name]
                    
                    # Determine image type based on expected label
                    if expected_label == 0:
                        image_type = 'original'
                    else:
                        image_type = 'adversarial'
                    
                    print(f"   Expected label: {expected_label} ({'clean' if expected_label == 0 else 'adversarial'})")
                    
                    # Run inference on external folder with explicit expected label
                    pred, prob, acc, detail = inference_detector(
                        model_path, folder_path, image_type, args.batch_size, expected_label=expected_label
                    )
                    
                    # Store results
                    inference_results[dataset_name] = {
                        image_type: detail
                    }
                    
                    # Save individual CSV results
                    folder_basename_csv = get_folder_basename(folder_path)
                    model_basename = get_folder_basename(model_path.replace('.pkl', ''))
                    
                    csv_file = os.path.join(results_dir, f'external_{model_basename}_{folder_basename_csv}_{image_type}.csv')
                    results_data = []
                    for i, (pred_val, prob_val, true_label, img_path) in enumerate(zip(
                        pred, prob, detail['true_labels'], detail['image_paths']
                    )):
                        results_data.append([i, os.path.basename(img_path), pred_val, prob_val, true_label])
                    
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['image_index', 'filename', 'predicted_label', 'adversarial_probability', 'true_label'])
                        writer.writerows(results_data)
                    
                    print(f"   Results saved to: {csv_file}")
                    
                except Exception as e:
                    print(f"Error during inference on {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"   No external folders provided - skipping external inference")
        
        # Generate final summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nTraining Results:")
        print(f"  Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
        print(f"  Validation Accuracy: {valid_acc:.4f}, AUC: {valid_auc:.4f}")
        
        print(f"\nInternal Test Split Results:")
        print(f"  Clean Images: {clean_accuracy:.4f} accuracy ({len(clean_indices)} samples)")
        print(f"  Adversarial Images: {adv_accuracy:.4f} accuracy ({len(adv_indices)} samples)")
        print(f"  Overall: {test_acc:.4f} accuracy")
        
        if has_external:
            print(f"\nExternal Dataset Results:")
            for dataset_name, results in inference_results.items():
                if dataset_name == 'internal_test':
                    continue
                    
                for img_type, detail in results.items():
                    accuracy = detail['accuracy']
                    confidence = detail['avg_confidence']
                    n_samples = detail['n_total']
                    print(f"  {dataset_name} ({img_type}): {accuracy:.4f} accuracy, {confidence:.4f} avg confidence ({n_samples} samples)")
        
        # Save comprehensive markdown results
        md_file = save_markdown_results(train_results, inference_results, model_info, folder_info)
        
        print(f"\nAll results saved!")
        print(f"  Model: {model_path}")
        print(f"  Results directory: {results_dir}/")
        print(f"  Detailed report: {md_file}")
        
    except Exception as e:
        print(f"Error training detector: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()