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
    """Dataset class for loading adversarial and benign images from pickle files"""
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

def load_pickle_data(clean_file_path, adv_file_path, max_samples=None):
    """
    Load clean and adversarial data from separate pickle files
    Based on the original format: (data, labels) = pickle.load(file)
    
    Args:
        clean_file_path: Path to clean images pickle file
        adv_file_path: Path to adversarial images pickle file
        max_samples: Maximum number of samples to load from each file (None for all)
    
    Returns:
        images: List of image tensors
        labels: List of labels (0=clean, 1=adversarial)
    """
    import pickle
    
    print(f"Loading clean data from: {clean_file_path}")
    print(f"Loading adversarial data from: {adv_file_path}")
    if max_samples is not None:
        print(f"Limiting to {max_samples} samples per file")
    
    all_images = []
    all_labels = []
    
    # Load clean data using pickle
    try:
        with open(clean_file_path, 'rb') as handle:
            clean_data, clean_original_labels = pickle.load(handle)
        print(f"Clean data type: {type(clean_data)}")
        print(f"Clean data shape: {clean_data.shape if hasattr(clean_data, 'shape') else 'No shape'}")
    except Exception as e:
        print(f"Error loading clean data: {e}")
        raise e
    
    # Load adversarial data using pickle
    try:
        with open(adv_file_path, 'rb') as handle:
            adv_data, adv_original_labels = pickle.load(handle)
        print(f"Adversarial data type: {type(adv_data)}")
        print(f"Adversarial data shape: {adv_data.shape if hasattr(adv_data, 'shape') else 'No shape'}")
    except Exception as e:
        print(f"Error loading adversarial data: {e}")
        raise e
    
    # Process clean images (convert to torch tensors and assign label 0)
    print("Processing clean images...")
    if isinstance(clean_data, np.ndarray):
        clean_data_torch = torch.from_numpy(clean_data)
        # Limit samples if specified
        num_clean = min(clean_data_torch.shape[0], max_samples) if max_samples is not None else clean_data_torch.shape[0]
        for i in tqdm(range(num_clean), desc="Processing clean images"):
            all_images.append(clean_data_torch[i])
            all_labels.append(0)  # Clean = 0
    elif isinstance(clean_data, list):
        # Handle list format
        num_clean = min(len(clean_data), max_samples) if max_samples is not None else len(clean_data)
        for i in tqdm(range(num_clean), desc="Processing clean images"):
            img = clean_data[i]
            if isinstance(img, np.ndarray):
                all_images.append(torch.from_numpy(img))
            else:
                all_images.append(img)
            all_labels.append(0)
    else:
        raise ValueError(f"Unexpected clean data format: {type(clean_data)}")
    
    # Process adversarial images (convert to torch tensors and assign label 1)
    print("Processing adversarial images...")
    if isinstance(adv_data, np.ndarray):
        adv_data_torch = torch.from_numpy(adv_data)
        # Limit samples if specified
        num_adv = min(adv_data_torch.shape[0], max_samples) if max_samples is not None else adv_data_torch.shape[0]
        for i in tqdm(range(num_adv), desc="Processing adversarial images"):
            all_images.append(adv_data_torch[i])
            all_labels.append(1)  # Adversarial = 1
    elif isinstance(adv_data, list):
        # Handle list format
        num_adv = min(len(adv_data), max_samples) if max_samples is not None else len(adv_data)
        for i in tqdm(range(num_adv), desc="Processing adversarial images"):
            img = adv_data[i]
            if isinstance(img, np.ndarray):
                all_images.append(torch.from_numpy(img))
            else:
                all_images.append(img)
            all_labels.append(1)
    else:
        raise ValueError(f"Unexpected adversarial data format: {type(adv_data)}")
    
    print(f"Loaded {len(all_images)} images total")
    print(f"Clean images: {all_labels.count(0)}")
    print(f"Adversarial images: {all_labels.count(1)}")
    
    return all_images, all_labels

def load_single_pickle_data(file_path, label):
    """
    Load data from a single pickle file and assign uniform labels
    
    Args:
        file_path: Path to pickle file
        label: Label to assign (0 for clean, 1 for adversarial)
    
    Returns:
        images: List of image tensors
        labels: List of labels (all same value)
    """
    print(f"Loading data from: {file_path} with label {label}")
    
    # Load data
    try:
        if torch.cuda.is_available():
            data = torch.load(file_path, map_location=device, weights_only=False)
        else:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory insufficient, loading to CPU instead...")
            data = torch.load(file_path, map_location='cpu', weights_only=False)
        else:
            raise e
    
    all_images = []
    all_labels = []
    
    # Process images
    label_name = "clean" if label == 0 else "adversarial"
    print(f"Processing {label_name} images...")
    
    if isinstance(data, torch.Tensor):
        # If it's a single tensor of shape [N, C, H, W]
        for i in tqdm(range(data.shape[0]), desc=f"Processing {label_name} images"):
            all_images.append(data[i])
            all_labels.append(label)
    elif isinstance(data, list):
        # If it's a list of tensors
        for img in tqdm(data, desc=f"Processing {label_name} images"):
            if len(img.shape) == 4:  # Batch dimension
                for i in range(img.shape[0]):
                    all_images.append(img[i])
                    all_labels.append(label)
            else:  # Single image
                all_images.append(img)
                all_labels.append(label)
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    print(f"Loaded {len(all_images)} {label_name} images")
    
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

def get_file_basename(file_path):
    """Extract basename from file path for naming saved files"""
    return os.path.splitext(os.path.basename(file_path))[0]

def train_detector(clean_train_file, adv_train_file, clean_valid_file, adv_valid_file, 
                  model_name='resnet50', batch_size=32):
    """
    Train a detector using separate clean and adversarial files
    
    Args:
        clean_train_file: Path to clean training data pickle file
        adv_train_file: Path to adversarial training data pickle file
        clean_valid_file: Path to clean validation data pickle file
        adv_valid_file: Path to adversarial validation data pickle file
        model_name: Feature extractor model ('resnet50' or 'vit')
        batch_size: Batch size for feature extraction
    """
    # Create identifier from training files
    clean_basename = get_file_basename(clean_train_file)
    adv_basename = get_file_basename(adv_train_file)
    train_identifier = f"{clean_basename}_{adv_basename}"
    
    print(f"\n{'='*60}")
    print(f"Training detector using {model_name}")
    print(f"Clean train: {clean_train_file}")
    print(f"Adv train: {adv_train_file}")
    print(f"Clean valid: {clean_valid_file}")
    print(f"Adv valid: {adv_valid_file}")
    print(f"{'='*60}")
    
    # Load training data
    train_images, train_labels = load_pickle_data(clean_train_file, adv_train_file)
    
    # Load validation data
    valid_images, valid_labels = load_pickle_data(clean_valid_file, adv_valid_file)
    
    # Create dataset with transforms for feature extraction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = AdvImageDataset(train_images, train_labels, transform=transform)
    valid_dataset = AdvImageDataset(valid_images, valid_labels, transform=transform)
    
    # Extract features
    feature_extractor = FeatureExtractor(model_name)
    
    print("Extracting training features...")
    train_features, train_labels = extract_features(train_dataset, feature_extractor, batch_size=batch_size)
    
    print("Extracting validation features...")
    valid_features, valid_labels = extract_features(valid_dataset, feature_extractor, batch_size=batch_size)
    
    # Create test split from training data (80/20)
    n_train = len(train_features)
    indices = list(range(n_train))
    random.shuffle(indices)
    
    test_size = int(n_train * 0.2)
    test_indices = indices[:test_size]
    actual_train_indices = indices[test_size:]
    
    # Create splits
    actual_train_features = train_features[actual_train_indices]
    actual_train_labels = train_labels[actual_train_indices]
    test_features = train_features[test_indices]
    test_labels = train_labels[test_indices]
    
    print(f"Data split:")
    print(f"  Training: {len(actual_train_features)} samples")
    print(f"  Testing: {len(test_features)} samples")
    print(f"  Validation: {len(valid_features)} samples")
    
    # Train SVM
    print("Training SVM classifier...")
    svm_model = SVC(probability=True, kernel='rbf', random_state=42, verbose=True)
    
    # Create a simple progress bar for SVM training
    print("  Starting SVM optimization...")
    start_time = time.time()
    svm_model.fit(actual_train_features, actual_train_labels)
    end_time = time.time()
    print(f"  SVM training completed in {end_time - start_time:.2f} seconds")
    
    # Save model and data
    save_name = os.path.join(model_dir, f'detector_{train_identifier}_{model_name}.pkl')
    torch.save({
        'model': svm_model,
        'train_data': (actual_train_features, actual_train_labels),
        'test_data': (test_features, test_labels),
        'valid_data': (valid_features, valid_labels),
        'feature_extractor': model_name,
        'clean_train_file': clean_train_file,
        'adv_train_file': adv_train_file,
        'clean_valid_file': clean_valid_file,
        'adv_valid_file': adv_valid_file
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
    
    print(f"\nResults for {train_identifier}:")
    print(f"  Class distribution - Test: {test_class_dist}, Valid: {valid_class_dist}")
    print(f"  Default threshold (0.5):")
    print(f"    Test  - Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
    print(f"    Valid - Accuracy: {valid_acc:.4f}, AUC: {valid_auc:.4f}")
    print(f"  Optimal threshold ({best_threshold:.3f}):")
    print(f"    Test  - Accuracy: {test_acc_optimal:.4f}")
    print(f"    Valid - Accuracy: {best_acc:.4f}")
    print(f"  Latency: {latency:.4f}s, Energy: {energy}")
    
    # Save results
    results = [[train_identifier, test_acc, test_auc, valid_acc, valid_auc, latency, energy]]
    os.makedirs(results_dir, exist_ok=True)
    np.savetxt(f'{results_dir}/detector_results_{train_identifier}_{model_name}.csv', 
               results, delimiter=',', fmt='%s',
               header='train_id,test_acc,test_auc,valid_acc,valid_auc,latency,energy')
    print(f"Results saved to {results_dir}/detector_results_{train_identifier}_{model_name}.csv")
    
    return train_identifier, test_acc, test_auc, valid_acc, valid_auc, latency, energy

def inference_detector(model_path, clean_file_path, adv_file_path, batch_size=32, max_samples=None):
    """
    Run inference on images using trained detector
    
    Args:
        model_path: Path to saved detector model (.pkl file)
        clean_file_path: Path to clean images pickle file
        adv_file_path: Path to adversarial images pickle file
        batch_size: Batch size for feature extraction
        max_samples: Maximum number of samples to process from each file (None for all)
    
    Returns:
        results for both clean and adversarial images
    """
    print(f"\n{'='*40}")
    print(f"Inference on clean and adversarial images")
    print(f"Clean file: {os.path.basename(clean_file_path)}")
    print(f"Adv file: {os.path.basename(adv_file_path)}")
    if max_samples is not None:
        print(f"Max samples per file: {max_samples}")
    print(f"{'='*40}")
    
    # Load trained model
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    svm_model = model_data['model']
    feature_extractor_name = model_data['feature_extractor']
    
    # Load inference data with sample limit
    images, labels = load_pickle_data(clean_file_path, adv_file_path, max_samples=max_samples)
    
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
    
    # Separate results by image type
    clean_indices = np.where(true_labels == 0)[0]
    adv_indices = np.where(true_labels == 1)[0]
    
    # Clean image results
    clean_pred = predictions[clean_indices]
    clean_prob = probabilities[clean_indices]
    clean_labels = true_labels[clean_indices]
    clean_acc = np.mean(clean_pred == clean_labels)
    
    # Adversarial image results
    adv_pred = predictions[adv_indices]
    adv_prob = probabilities[adv_indices]
    adv_labels_true = true_labels[adv_indices]
    adv_acc = np.mean(adv_pred == adv_labels_true)
    
    # Overall accuracy
    overall_acc = np.mean(predictions == true_labels)
    
    print(f"  Clean images: {len(clean_indices)} total")
    print(f"    Correctly classified as clean: {np.sum(clean_pred == 0)} ({np.sum(clean_pred == 0)/len(clean_indices):.1%})")
    print(f"    False positives (predicted as adv): {np.sum(clean_pred == 1)} ({np.sum(clean_pred == 1)/len(clean_indices):.1%})")
    print(f"    Accuracy: {clean_acc:.4f}")
    print(f"    Average adversarial probability: {np.mean(clean_prob):.4f}")
    
    print(f"  Adversarial images: {len(adv_indices)} total")
    print(f"    Correctly detected as adversarial: {np.sum(adv_pred == 1)} ({np.sum(adv_pred == 1)/len(adv_indices):.1%})")
    print(f"    Missed (predicted as clean): {np.sum(adv_pred == 0)} ({np.sum(adv_pred == 0)/len(adv_indices):.1%})")
    print(f"    Accuracy: {adv_acc:.4f}")
    print(f"    Average adversarial probability: {np.mean(adv_prob):.4f}")
    
    print(f"  Overall accuracy: {overall_acc:.4f}")
    print(f"  Inference time: {inference_time:.4f}s ({inference_time/len(features)*1000:.2f}ms per image)")
    
    # Create detailed results
    detailed_results = {
        'clean': {
            'n_total': len(clean_indices),
            'accuracy': clean_acc,
            'avg_confidence': np.mean(clean_prob),
            'predictions': clean_pred,
            'probabilities': clean_prob,
            'true_labels': clean_labels
        },
        'adv': {
            'n_total': len(adv_indices),
            'accuracy': adv_acc,
            'avg_confidence': np.mean(adv_prob),
            'predictions': adv_pred,
            'probabilities': adv_prob,
            'true_labels': adv_labels_true
        },
        'overall': {
            'n_total': len(features),
            'accuracy': overall_acc,
            'inference_time': inference_time,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels
        }
    }
    
    return detailed_results

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
    md_file = os.path.join(results_dir, f'results_{train_results["train_identifier"]}_{model_info["model"]}_{timestamp}.md')
    
    with open(md_file, 'w') as f:
        f.write(f"# Adversarial Detector Results\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Information
        f.write(f"## Model Information\n\n")
        f.write(f"- **Feature Extractor:** {model_info['model']}\n")
        f.write(f"- **Clean Training File:** {model_info['clean_train_file']}\n")
        f.write(f"- **Adversarial Training File:** {model_info['adv_train_file']}\n")
        f.write(f"- **Clean Validation File:** {model_info['clean_valid_file']}\n")
        f.write(f"- **Adversarial Validation File:** {model_info['adv_valid_file']}\n")
        f.write(f"- **Batch Size:** {model_info['batch_size']}\n\n")
        
        # File Information
        if file_info:
            f.write(f"## Test File Information\n\n")
            f.write(f"| File Type | Clean Path | Adv Path | Exists | Total Images |\n")
            f.write(f"|-----------|------------|----------|--------|-------------|\n")
            for name, info in file_info.items():
                if info['provided']:
                    clean_exists = "✓" if info['clean_exists'] else "✗"
                    adv_exists = "✓" if info['adv_exists'] else "✗"
                    exists_str = f"{clean_exists}/{adv_exists}"
                    length_str = str(info['length']) if info['length'] is not None else "Error"
                    f.write(f"| {name.replace('_', ' ').title()} | `{info['clean_path']}` | `{info['adv_path']}` | {exists_str} | {length_str} |\n")
                else:
                    f.write(f"| {name.replace('_', ' ').title()} | Not provided | Not provided | - | - |\n")
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
            
            clean_count = results['clean']['n_total']
            adv_count = results['adv']['n_total']
            
            f.write(f"**Sample Counts:** {clean_count} clean, {adv_count} adversarial\n\n")
            
            f.write(f"| Image Type | Accuracy | Avg Confidence | Details |\n")
            f.write(f"|------------|----------|----------------|----------|\n")
            f.write(f"| Clean | {results['clean']['accuracy']:.4f} | {results['clean']['avg_confidence']:.4f} | Should be classified as clean (0) |\n")
            f.write(f"| Adversarial | {results['adv']['accuracy']:.4f} | {results['adv']['avg_confidence']:.4f} | Should be detected as adversarial (1) |\n")
            f.write(f"| Overall | {results['overall']['accuracy']:.4f} | - | Combined accuracy |\n\n")
            
            # Additional metrics
            clean_acc = results['clean']['accuracy']
            adv_acc = results['adv']['accuracy']
            f.write(f"**Detection Summary:**\n")
            f.write(f"- Clean Classification Rate: {clean_acc:.1%}\n")
            f.write(f"- Adversarial Detection Rate: {adv_acc:.1%}\n")
            f.write(f"- False Positive Rate: {1-clean_acc:.1%}\n")
            f.write(f"- False Negative Rate: {1-adv_acc:.1%}\n")
            f.write(f"- Overall Balance: {(clean_acc + adv_acc)/2:.4f}\n")
            f.write(f"- Inference Time: {results['overall']['inference_time']:.4f}s\n\n")
        
        # Summary
        f.write(f"## Summary\n\n")
        f.write(f"This detector was trained on {train_results['train_identifier']} and evaluated on multiple test sets. ")
        f.write(f"The model achieved {train_results['test_acc']:.1%} test accuracy during training and showed ")
        f.write(f"varying performance across different attack types during inference.\n\n")
        
        f.write(f"**Key Findings:**\n")
        for dataset_name, results in inference_results.items():
            clean_rate = results['clean']['accuracy']
            adv_rate = results['adv']['accuracy']
            f.write(f"- {dataset_name}: {clean_rate:.1%} clean classification, {adv_rate:.1%} adversarial detection\n")
    
    print(f"Detailed results saved to: {md_file}")
    return md_file

def main():
    """
    Main function to train detector and run comprehensive inference
    """
    parser = argparse.ArgumentParser(description='Train adversarial detector using separate clean/adversarial files')
    
    # Training files
    parser.add_argument('--clean_train', type=str, required=True,
                        help='Path to clean training data pickle file')
    parser.add_argument('--adv_train', type=str, required=True,
                        help='Path to adversarial training data pickle file')
    parser.add_argument('--clean_valid', type=str, required=True,
                        help='Path to clean validation data pickle file')
    parser.add_argument('--adv_valid', type=str, required=True,
                        help='Path to adversarial validation data pickle file')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'vit'],
                        help='Feature extractor model (default: resnet50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction (default: 32)')
    
    # Additional test files (optional)
    parser.add_argument('--clean_jpeg', type=str,
                        help='Path to clean JPEG data pickle file (optional)')
    parser.add_argument('--adv_jpeg', type=str,
                        help='Path to adversarial JPEG data pickle file (optional)')
    
    parser.add_argument('--clean_spatial', type=str,
                        help='Path to clean spatial data pickle file (optional)')
    parser.add_argument('--adv_spatial', type=str,
                        help='Path to adversarial spatial data pickle file (optional)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check if training files exist
    required_files = [args.clean_train, args.adv_train, args.clean_valid, args.adv_valid]
    required_names = ['clean_train', 'adv_train', 'clean_valid', 'adv_valid']
    
    for file_path, name in zip(required_files, required_names):
        if not os.path.exists(file_path):
            print(f"Error: Required file {name} ({file_path}) does not exist!")
            return
    
    # Training phase
    print("Starting training phase...")
    print(f"Clean training file: {args.clean_train}")
    print(f"Adversarial training file: {args.adv_train}")
    print(f"Clean validation file: {args.clean_valid}")
    print(f"Adversarial validation file: {args.adv_valid}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        train_identifier, test_acc, test_auc, valid_acc, valid_auc, latency, energy = train_detector(
            args.clean_train, 
            args.adv_train,
            args.clean_valid,
            args.adv_valid,
            model_name=args.model, 
            batch_size=args.batch_size
        )
        
        print(f"\nTraining Summary:")
        print(f"Train ID: {train_identifier}")
        print(f"Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
        print(f"Valid Accuracy: {valid_acc:.4f}, AUC: {valid_auc:.4f}")
        print(f"Latency: {latency:.4f}s, Energy: {energy}")
        
        # Prepare training results for markdown
        train_results = {
            'train_identifier': train_identifier,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'valid_acc': valid_acc,
            'valid_auc': valid_auc,
            'latency': latency,
            'energy': energy
        }
        
        model_info = {
            'model': args.model,
            'clean_train_file': os.path.basename(args.clean_train),
            'adv_train_file': os.path.basename(args.adv_train),
            'clean_valid_file': os.path.basename(args.clean_valid),
            'adv_valid_file': os.path.basename(args.adv_valid),
            'batch_size': args.batch_size
        }
        
        # Inference phase - validation files are used as test set
        # Additional datasets for comparison
        additional_file_pairs = {
            'jpeg': (args.clean_jpeg, args.adv_jpeg),
            'spatial': (args.clean_spatial, args.adv_spatial)
        }
        
        # Debug: Show what files were provided
        print(f"\nProvided additional test file pairs:")
        for name, (clean_path, adv_path) in additional_file_pairs.items():
            if clean_path is not None and adv_path is not None:
                clean_exists = os.path.exists(clean_path)
                adv_exists = os.path.exists(adv_path)
                print(f"  {name}:")
                print(f"    Clean: {clean_path} (exists: {clean_exists})")
                print(f"    Adv: {adv_path} (exists: {adv_exists})")
                
                if clean_exists and adv_exists:
                    try:
                        # Count images in both files using correct pickle format
                        import pickle
                        
                        with open(clean_path, 'rb') as handle:
                            clean_data, clean_labels = pickle.load(handle)
                        with open(adv_path, 'rb') as handle:
                            adv_data, adv_labels = pickle.load(handle)
                        
                        # Count clean images
                        if isinstance(clean_data, np.ndarray):
                            clean_count = clean_data.shape[0]
                        elif isinstance(clean_data, list):
                            clean_count = len(clean_data)
                        else:
                            clean_count = "Unknown format"
                        
                        # Count adversarial images
                        if isinstance(adv_data, np.ndarray):
                            adv_count = adv_data.shape[0]
                        elif isinstance(adv_data, list):
                            adv_count = len(adv_data)
                        else:
                            adv_count = "Unknown format"
                        
                        total_count = clean_count + adv_count if isinstance(clean_count, int) and isinstance(adv_count, int) else "Error"
                        print(f"    -> Clean: {clean_count}, Adv: {adv_count}, Total: {total_count}")
                        
                    except Exception as e:
                        print(f"    -> Error reading files: {e}")
                else:
                    if not clean_exists:
                        print(f"    -> Clean file NOT FOUND!")
                    if not adv_exists:
                        print(f"    -> Adversarial file NOT FOUND!")
            else:
                missing = []
                if clean_path is None:
                    missing.append("clean")
                if adv_path is None:
                    missing.append("adversarial")
                print(f"  {name}: Missing {'/'.join(missing)} file(s)")
        
        # Remove pairs where either file is None or doesn't exist
        # Always include validation as test set
        valid_test_pairs = {
            'validation_as_test': (args.clean_valid, args.adv_valid)
        }
        file_info = {}
        
        # Add validation as test to file_info
        try:
            import pickle
            
            with open(args.clean_valid, 'rb') as handle:
                clean_data, clean_labels = pickle.load(handle)
            with open(args.adv_valid, 'rb') as handle:
                adv_data, adv_labels = pickle.load(handle)
            
            if isinstance(clean_data, np.ndarray):
                clean_count = clean_data.shape[0]
            elif isinstance(clean_data, list):
                clean_count = len(clean_data)
            else:
                clean_count = 0
            
            if isinstance(adv_data, np.ndarray):
                adv_count = adv_data.shape[0]
            elif isinstance(adv_data, list):
                adv_count = len(adv_data)
            else:
                adv_count = 0
            
            total_count = clean_count + adv_count
        except:
            total_count = "Error"
        
        file_info['validation_as_test'] = {
            'provided': True,
            'clean_path': args.clean_valid,
            'adv_path': args.adv_valid,
            'clean_exists': True,
            'adv_exists': True,
            'length': total_count
        }
        
        # Process additional file pairs
        for name, (clean_path, adv_path) in additional_file_pairs.items():
            if clean_path is not None and adv_path is not None:
                clean_exists = os.path.exists(clean_path)
                adv_exists = os.path.exists(adv_path)
                
                if clean_exists and adv_exists:
                    valid_test_pairs[name] = (clean_path, adv_path)
                    
                    # Calculate total images for file_info
                    try:
                        import pickle
                        
                        with open(clean_path, 'rb') as handle:
                            clean_data, clean_labels = pickle.load(handle)
                        with open(adv_path, 'rb') as handle:
                            adv_data, adv_labels = pickle.load(handle)
                        
                        if isinstance(clean_data, np.ndarray):
                            clean_count = clean_data.shape[0]
                        elif isinstance(clean_data, list):
                            clean_count = len(clean_data)
                        else:
                            clean_count = 0
                        
                        if isinstance(adv_data, np.ndarray):
                            adv_count = adv_data.shape[0]
                        elif isinstance(adv_data, list):
                            adv_count = len(adv_data)
                        else:
                            adv_count = 0
                        
                        total_count = clean_count + adv_count
                    except:
                        total_count = "Error"
                else:
                    total_count = None
                
                file_info[name] = {
                    'provided': True,
                    'clean_path': clean_path,
                    'adv_path': adv_path,
                    'clean_exists': clean_exists,
                    'adv_exists': adv_exists,
                    'length': total_count
                }
            else:
                file_info[name] = {
                    'provided': False,
                    'clean_path': clean_path,
                    'adv_path': adv_path,
                    'clean_exists': False,
                    'adv_exists': False,
                    'length': None
                }
        
        print(f"\nTest sets to process: {list(valid_test_pairs.keys())}")
        
        # Always run inference since we have validation data as test set
        print(f"\n{'='*60}")
        print("Starting comprehensive inference phase...")
        print("Using validation data as test set + additional datasets if provided")
        print(f"{'='*60}")
        
        # Construct model path from training
        model_path = os.path.join(model_dir, f'detector_{train_identifier}_{args.model}.pkl')
        inference_results = {}
        
        for dataset_name, (clean_file, adv_file) in valid_test_pairs.items():
            print(f"\n--- Processing {dataset_name} ---")
            print(f"Clean file: {clean_file}")
            print(f"Adv file: {adv_file}")
            
            try:
                # Run inference on both clean and adversarial images
                detailed_results = inference_detector(
                    model_path, clean_file, adv_file, args.batch_size, max_samples=1000
                )
                
                # Store results
                inference_results[dataset_name] = detailed_results
                
                # Save individual CSV results
                clean_basename = get_file_basename(clean_file)
                adv_basename = get_file_basename(adv_file)
                model_basename = get_file_basename(model_path)
                inference_basename = f"{clean_basename}_{adv_basename}"
                
                # Save combined results
                combined_file = os.path.join(results_dir, f'inference_{model_basename}_{inference_basename}_combined.csv')
                combined_results = np.column_stack([
                    range(len(detailed_results['overall']['predictions'])), 
                    detailed_results['overall']['predictions'], 
                    detailed_results['overall']['probabilities'], 
                    detailed_results['overall']['true_labels']
                ])
                np.savetxt(combined_file, combined_results, delimiter=',', fmt='%d,%d,%.6f,%d',
                            header='image_index,predicted_label,adversarial_probability,true_label')
                
                # Save clean results separately
                clean_file_out = os.path.join(results_dir, f'inference_{model_basename}_{inference_basename}_clean.csv')
                clean_results = np.column_stack([
                    range(len(detailed_results['clean']['predictions'])), 
                    detailed_results['clean']['predictions'], 
                    detailed_results['clean']['probabilities'], 
                    detailed_results['clean']['true_labels']
                ])
                np.savetxt(clean_file_out, clean_results, delimiter=',', fmt='%d,%d,%.6f,%d',
                            header='image_index,predicted_label,adversarial_probability,true_label')
                
                # Save adversarial results separately
                adv_file_out = os.path.join(results_dir, f'inference_{model_basename}_{inference_basename}_adv.csv')
                adv_results = np.column_stack([
                    range(len(detailed_results['adv']['predictions'])), 
                    detailed_results['adv']['predictions'], 
                    detailed_results['adv']['probabilities'], 
                    detailed_results['adv']['true_labels']
                ])
                np.savetxt(adv_file_out, adv_results, delimiter=',', fmt='%d,%d,%.6f,%d',
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
                clean_acc = results['clean']['accuracy']
                adv_acc = results['adv']['accuracy']
                overall_acc = results['overall']['accuracy']
                clean_conf = results['clean']['avg_confidence']
                adv_conf = results['adv']['avg_confidence']
                
                print(f"\n{dataset_name.upper()}:")
                print(f"  Clean Classification: {clean_acc:.4f} (confidence: {clean_conf:.4f})")
                print(f"  Adversarial Detection: {adv_acc:.4f} (confidence: {adv_conf:.4f})")
                print(f"  Overall Accuracy: {overall_acc:.4f}")
                print(f"  Balance Score: {(clean_acc + adv_acc)/2:.4f}")
            
            # Save markdown results
            md_file = save_markdown_results(train_results, inference_results, model_info, file_info)
        
    except Exception as e:
        print(f"Error during training/inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()