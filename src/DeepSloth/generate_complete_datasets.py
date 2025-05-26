#!/usr/bin/env python3
"""
Generate COMPLETE training and validation datasets for adversarial training.
MODIFIED: Uses only 1000 samples for all datasets.

This script creates ALL possible combinations:
1. Clean training data (1000 samples)
2. Clean validation data (1000 samples)
3. Adversarial training data (1000 samples, per-sample attacks)
4. Adversarial validation data (1000 samples, per-sample attacks)
5. Universal perturbation (learned from 1000 training samples)
6. Universal adversarial training data (universal perturbation applied to 1000 train samples)
7. Universal adversarial validation data (universal perturbation applied to 1000 valid samples)
"""

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Import from the original codebase
import models
from datasets import load_train_loader, load_valid_loader
import attacks.ours_l2 as ours_l2
import attacks.ours_l1 as ours_l1
import attacks.ours_linf as ours_linf

def save_dataset(data, labels, filepath, description):
    """Save dataset with logging"""
    print(f"Saving {description} to {filepath}")
    print(f"  Data shape: {data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    with open(filepath, 'wb') as f:
        pickle.dump((data, labels), f, protocol=4)
    
    file_size = os.path.getsize(filepath) / (1024*1024)
    print(f"  File size: {file_size:.2f} MB")

def save_perturbation(perturbation, filepath, description):
    """Save perturbation with logging"""
    print(f"Saving {description} to {filepath}")
    print(f"  Perturbation shape: {perturbation.shape}")
    print(f"  Perturbation range: [{perturbation.min():.6f}, {perturbation.max():.6f}]")
    
    # Handle both torch tensors and numpy arrays
    if isinstance(perturbation, torch.Tensor):
        norm_value = torch.norm(perturbation).item()
    else:  # numpy array
        norm_value = np.linalg.norm(perturbation)
    
    print(f"  Perturbation norm: {norm_value:.6f}")
    
    with open(filepath, 'wb') as f:
        pickle.dump(perturbation, f, protocol=4)
    
    file_size = os.path.getsize(filepath) / (1024*1024)
    print(f"  File size: {file_size:.2f} MB")

def extract_clean_data(data_loader, max_samples=1000, use_cuda=False, desc="Processing data"):
    """Extract clean data and labels from a data loader (limited to max_samples)"""
    data_list, labels_list = [], []
    total_samples = 0
    
    for batch_idx, (data, labels) in enumerate(tqdm(data_loader, desc=desc)):
        if use_cuda:
            data, labels = data.cuda(), labels.cuda()
        
        # Check if we need to truncate this batch
        batch_size = data.size(0)
        if total_samples + batch_size > max_samples:
            # Take only the samples we need to reach max_samples
            needed_samples = max_samples - total_samples
            data = data[:needed_samples]
            labels = labels[:needed_samples]
            batch_size = needed_samples
        
        data_list.append(data.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        total_samples += batch_size
        
        # Stop if we've reached the maximum
        if total_samples >= max_samples:
            break
    
    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    print(f"  Extracted {len(all_data)} samples (requested max: {max_samples})")
    return all_data, all_labels

def create_limited_data_loader(data_loader, max_samples=1000, use_cuda=False):
    """Create a limited dataset from the data loader and return as a new data loader"""
    data_list, labels_list = [], []
    total_samples = 0
    
    for batch_idx, (data, labels) in enumerate(data_loader):
        # Check if we need to truncate this batch
        batch_size = data.size(0)
        if total_samples + batch_size > max_samples:
            # Take only the samples we need to reach max_samples
            needed_samples = max_samples - total_samples
            data = data[:needed_samples]
            labels = labels[:needed_samples]
            batch_size = needed_samples
        
        data_list.append(data)
        labels_list.append(labels)
        total_samples += batch_size
        
        # Stop if we've reached the maximum
        if total_samples >= max_samples:
            break
    
    print(f"  Created limited dataset with {total_samples} samples")
    
    # Create a new data loader from the limited data
    class LimitedDataLoader:
        def __init__(self, data_list, labels_list):
            self.data_list = data_list
            self.labels_list = labels_list
        
        def __iter__(self):
            for data, labels in zip(self.data_list, self.labels_list):
                yield data, labels
        
        def __len__(self):
            return len(self.data_list)
    
    return LimitedDataLoader(data_list, labels_list)

def apply_universal_perturbation_limited(limited_loader, perturbation, use_cuda=False, desc="Applying universal perturbation"):
    """Apply universal perturbation to limited data"""
    result_data_list, result_labels_list = [], []
    
    # Convert perturbation to torch tensor if it's numpy, and move to appropriate device
    if isinstance(perturbation, np.ndarray):
        perturbation = torch.from_numpy(perturbation)
    
    if use_cuda:
        perturbation = perturbation.cuda()
    
    for data, labels in tqdm(limited_loader, desc=desc):
        if use_cuda:
            data, labels = data.cuda(), labels.cuda()
        
        # Apply universal perturbation
        perturbed_data = data + perturbation
        # Clamp to valid range [0, 1] (assuming normalized data)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        result_data_list.append(perturbed_data.cpu().numpy())
        result_labels_list.append(labels.cpu().numpy())
    
    all_data = np.concatenate(result_data_list, axis=0)
    all_labels = np.concatenate(result_labels_list, axis=0)
    
    return all_data, all_labels

def craft_adversarial_limited(attack_func, model, limited_loader, device='cpu', **attack_kwargs):
    """Craft adversarial examples for limited data"""
    total_samples = sum(len(labels) for _, labels in limited_loader)
    print(f"  Crafting adversarial examples for {total_samples} samples")
    
    # Call the attack function directly with the limited loader
    adv_data, adv_labels = attack_func(model, limited_loader, device=device, **attack_kwargs)
    
    return adv_data, adv_labels

def generate_complete_datasets(args, model, use_cuda=False):
    """Generate all combinations of clean and adversarial datasets (1000 samples each)"""
    
    # Load data loaders
    train_loader = load_train_loader(args.dataset, nbatch=args.batch_size)
    valid_loader = load_valid_loader(args.dataset, nbatch=args.batch_size)
    
    save_folder = os.path.join('complete_datasets', args.dataset, 
                              f'{args.dataset}_{args.network}_{args.nettype}_1000samples')
    os.makedirs(save_folder, exist_ok=True)
    
    print(f"Saving all datasets to: {save_folder}")
    print(f"Using 1000 samples for all datasets")
    
    # ========================================================================
    # STEP 1: Generate Clean Datasets (1000 samples each)
    # ========================================================================
    
    print("\n" + "="*60)
    print("STEP 1: Generating Clean Datasets (1000 samples each)")
    print("="*60)
    
    # Clean training data
    print("\n--- Clean Training Data ---")
    clean_train_data, clean_train_labels = extract_clean_data(
        train_loader, max_samples=1000, use_cuda=use_cuda, 
        desc="Extracting clean training data (1000 samples)"
    )
    save_dataset(clean_train_data, clean_train_labels,
                os.path.join(save_folder, 'clean_train_1000.pickle'),
                "Clean training dataset (1000 samples)")
    
    # Clean validation data
    print("\n--- Clean Validation Data ---")
    clean_valid_data, clean_valid_labels = extract_clean_data(
        valid_loader, max_samples=1000, use_cuda=use_cuda,
        desc="Extracting clean validation data (1000 samples)"
    )
    save_dataset(clean_valid_data, clean_valid_labels,
                os.path.join(save_folder, 'clean_valid_1000.pickle'),
                "Clean validation dataset (1000 samples)")
    
    # ========================================================================
    # STEP 2: Generate Per-Sample Adversarial Datasets (1000 samples each)
    # ========================================================================
    
    print("\n" + "="*60)
    print("STEP 2: Generating Per-Sample Adversarial Datasets (1000 samples each)")
    print("="*60)
    
    # Set attack parameters based on norm and dataset
    if args.ellnorm == 'l2':
        gamma = 0.05 if args.dataset == 'tinyimagenet' else 0.1
        attack_func = ours_l2.craft_per_sample_perturb_attack
        univ_func = ours_l2.craft_universal_perturb_attack
        attack_kwargs = {'gamma': gamma}
    elif args.ellnorm == 'l1':
        epsilon = 16 if args.dataset == 'tinyimagenet' else 8
        epsstep = 1.0 if args.dataset == 'tinyimagenet' else 0.5
        attack_func = ours_l1.craft_per_sample_perturb_attack
        univ_func = ours_l1.craft_universal_perturb_attack
        attack_kwargs = {'epsilon': epsilon, 'eps_step': epsstep}
        univ_kwargs = {'epsilon': epsilon, 'step_size': epsstep}
    elif args.ellnorm == 'linf':
        attack_func = ours_linf.craft_per_sample_perturb_attack
        univ_func = ours_linf.craft_universal_perturb_attack
        attack_kwargs = {}
        univ_kwargs = {}
    else:
        raise ValueError(f"Unsupported norm: {args.ellnorm}")
    
    device = 'cuda' if use_cuda else 'cpu'
    
    # Create limited training data for adversarial attacks
    print("\n--- Preparing Limited Training Data for Attacks ---")
    train_limited_loader = create_limited_data_loader(
        train_loader, max_samples=1000, use_cuda=use_cuda
    )
    
    # Create limited validation data for adversarial attacks
    print("\n--- Preparing Limited Validation Data for Attacks ---")
    valid_limited_loader = create_limited_data_loader(
        valid_loader, max_samples=1000, use_cuda=use_cuda
    )
    
    # Per-sample adversarial training data
    print(f"\n--- Per-Sample Adversarial Training Data ({args.ellnorm}) ---")
    adv_train_data, adv_train_labels = craft_adversarial_limited(
        attack_func, model, train_limited_loader,
        device=device, **attack_kwargs
    )
    # Take the final iteration (most adversarial)
    final_adv_train = adv_train_data[-1]
    save_dataset(final_adv_train, adv_train_labels,
                os.path.join(save_folder, f'adv_persample_train_{args.ellnorm}_1000.pickle'),
                f"Per-sample adversarial training dataset ({args.ellnorm}, 1000 samples)")
    
    # Per-sample adversarial validation data
    print(f"\n--- Per-Sample Adversarial Validation Data ({args.ellnorm}) ---")
    adv_valid_data, adv_valid_labels = craft_adversarial_limited(
        attack_func, model, valid_limited_loader,
        device=device, **attack_kwargs
    )
    # Take the final iteration (most adversarial)
    final_adv_valid = adv_valid_data[-1]
    save_dataset(final_adv_valid, adv_valid_labels,
                os.path.join(save_folder, f'adv_persample_valid_{args.ellnorm}_1000.pickle'),
                f"Per-sample adversarial validation dataset ({args.ellnorm}, 1000 samples)")
    
    # ========================================================================
    # STEP 3: Generate Universal Perturbation (from 1000 training samples)
    # ========================================================================
    
    print("\n" + "="*60)
    print("STEP 3: Generating Universal Perturbation (from 1000 training samples)")
    print("="*60)
    
    print(f"\n--- Universal Perturbation ({args.ellnorm}) ---")
    if args.ellnorm == 'l1':
        # Universal attack functions return only perturbations, not (data, labels)
        universal_perturbations = univ_func(
            model, train_limited_loader, device=device, **univ_kwargs
        )
    else:
        # Universal attack functions return only perturbations, not (data, labels)
        universal_perturbations = univ_func(
            model, train_limited_loader, device=device, **attack_kwargs
        )
    
    # Take the final universal perturbation
    final_universal_perturbation = universal_perturbations[-1]
    save_perturbation(final_universal_perturbation,
                     os.path.join(save_folder, f'universal_perturbation_{args.ellnorm}_1000.pickle'),
                     f"Universal perturbation ({args.ellnorm}, from 1000 samples)")
    
    # ========================================================================
    # STEP 4: Generate Universal Adversarial Datasets (1000 samples each)
    # ========================================================================
    
    print("\n" + "="*60)
    print("STEP 4: Generating Universal Adversarial Datasets (1000 samples each)")
    print("="*60)
    
    # Universal adversarial training data
    print(f"\n--- Universal Adversarial Training Data ({args.ellnorm}) ---")
    univ_adv_train_data, univ_adv_train_labels = apply_universal_perturbation_limited(
        train_limited_loader, final_universal_perturbation, use_cuda,
        "Applying universal perturbation to training data (1000 samples)"
    )
    save_dataset(univ_adv_train_data, univ_adv_train_labels,
                os.path.join(save_folder, f'adv_universal_train_{args.ellnorm}_1000.pickle'),
                f"Universal adversarial training dataset ({args.ellnorm}, 1000 samples)")
    
    # Universal adversarial validation data
    print(f"\n--- Universal Adversarial Validation Data ({args.ellnorm}) ---")
    univ_adv_valid_data, univ_adv_valid_labels = apply_universal_perturbation_limited(
        valid_limited_loader, final_universal_perturbation, use_cuda,
        "Applying universal perturbation to validation data (1000 samples)"
    )
    save_dataset(univ_adv_valid_data, univ_adv_valid_labels,
                os.path.join(save_folder, f'adv_universal_valid_{args.ellnorm}_1000.pickle'),
                f"Universal adversarial validation dataset ({args.ellnorm}, 1000 samples)")
    
    return save_folder

def main():
    parser = argparse.ArgumentParser(description='Generate COMPLETE adversarial training datasets (1000 samples each)')
    
    # Basic configurations
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset name (cifar10 or tinyimagenet)')
    parser.add_argument('--network', type=str, default='resnet56',
                        help='Network architecture')
    parser.add_argument('--nettype', type=str, default='sdn_ic_only',
                        help='Network type')
    parser.add_argument('--ellnorm', type=str, default='l2',
                        help='Attack norm (l1, l2, or linf)')
    parser.add_argument('--batch-size', type=int, default=250,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPLETE ADVERSARIAL DATASET GENERATION (1000 SAMPLES)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Network: {args.network}_{args.nettype}")
    print(f"Attack norm: {args.ellnorm}")
    print(f"Batch size: {args.batch_size}")
    print(f"Samples per dataset: 1000")
    
    # Setup CUDA
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    
    # Load model
    netpath = os.path.join('models', args.dataset)
    netname = f'{args.dataset}_{args.network}_{args.nettype}'
    model, params = models.load_model(netpath, netname, epoch=-1)
    if use_cuda:
        model.cuda()
    model.eval()
    print(f"Loaded model: {netname}")
    
    # Generate all datasets
    save_folder = generate_complete_datasets(args, model, use_cuda)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"All datasets saved to: {save_folder}")
    print("\nGenerated files:")
    
    total_size = 0
    files_info = []
    
    for file in sorted(os.listdir(save_folder)):
        if file.endswith('.pickle'):
            filepath = os.path.join(save_folder, file)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            total_size += size_mb
            files_info.append((file, size_mb))
    
    # Print file summary
    for filename, size_mb in files_info:
        file_type = ""
        if "clean" in filename:
            file_type = "🟢 CLEAN"
        elif "persample" in filename:
            file_type = "🔴 PER-SAMPLE ADV"
        elif "universal_perturbation" in filename:
            file_type = "🟡 UNIVERSAL PERT"
        elif "universal" in filename:
            file_type = "🟠 UNIVERSAL ADV"
        
        print(f"  {file_type:<20} {filename:<45} ({size_mb:>6.2f} MB)")
    
    print(f"\nTotal size: {total_size:.2f} MB")
    
    print("\n" + "="*70)
    print("WHAT YOU NOW HAVE (1000 samples each):")
    print("="*70)
    print("✅ Clean training data (1000 samples)")
    print("✅ Clean validation data (1000 samples)")
    print("✅ Per-sample adversarial training data (1000 samples)")
    print("✅ Per-sample adversarial validation data (1000 samples)")
    print("✅ Universal perturbation (learned from 1000 training samples)")
    print("✅ Universal adversarial training data (1000 samples)")
    print("✅ Universal adversarial validation data (1000 samples)")
    print("\nYou can now train models with ANY combination of these datasets!")
    print("Note: All datasets are limited to 1000 samples for faster processing.")

if __name__ == "__main__":
    main()