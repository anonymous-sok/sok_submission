#!/usr/bin/env python3
"""
Complete dataset loader utilities for ALL generated datasets.
Handles clean, per-sample adversarial, universal perturbation, and universal adversarial data.
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class PickleDataset(Dataset):
    """PyTorch Dataset class for loading pickled data"""
    
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: numpy array of shape (N, C, H, W)
            labels: numpy array of shape (N,)
            transform: optional transform to apply to data
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()
            
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()
            
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

class CompleteDatasetManager:
    """Manager class for loading and handling ALL generated datasets"""
    
    def __init__(self, dataset_name, network, nettype, norm='l2', base_path='complete_datasets'):
        """
        Args:
            dataset_name: Name of dataset (e.g., 'cifar10')
            network: Network name (e.g., 'resnet56')
            nettype: Network type (e.g., 'sdn_ic_only')
            norm: Attack norm used (e.g., 'l2', 'l1', 'linf')
            base_path: Base directory where datasets are stored
        """
        self.dataset_name = dataset_name
        self.network = network
        self.nettype = nettype
        self.norm = norm
        
        self.dataset_folder = os.path.join(
            base_path, dataset_name, f'{dataset_name}_{network}_{nettype}'
        )
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_folder}")
        
        print(f"Dataset folder: {self.dataset_folder}")
        self._list_available_files()
    
    def _list_available_files(self):
        """List all available dataset files"""
        self.available_files = {}
        
        for file in os.listdir(self.dataset_folder):
            if file.endswith('.pickle'):
                file_path = os.path.join(self.dataset_folder, file)
                size_mb = os.path.getsize(file_path) / (1024*1024)
                
                # Categorize files
                category = "Unknown"
                if "clean" in file:
                    category = "Clean"
                elif "persample" in file:
                    category = "Per-sample Adversarial"
                elif "universal_perturbation" in file:
                    category = "Universal Perturbation"
                elif "universal" in file:
                    category = "Universal Adversarial"
                
                self.available_files[file] = {
                    'path': file_path,
                    'size_mb': size_mb,
                    'category': category
                }
        
        # Print organized file list
        print("\nAvailable datasets:")
        categories = ["Clean", "Per-sample Adversarial", "Universal Perturbation", "Universal Adversarial"]
        
        for category in categories:
            files_in_category = [(f, info) for f, info in self.available_files.items() 
                               if info['category'] == category]
            if files_in_category:
                print(f"\n  {category}:")
                for filename, info in files_in_category:
                    print(f"    {filename} ({info['size_mb']:.2f} MB)")
    
    def load_dataset(self, dataset_type):
        """
        Load a specific dataset
        
        Args:
            dataset_type: One of:
                - 'clean_train', 'clean_valid'
                - 'adv_persample_train_{norm}', 'adv_persample_valid_{norm}'
                - 'adv_universal_train_{norm}', 'adv_universal_valid_{norm}'
        
        Returns:
            tuple: (data, labels) as numpy arrays
        """
        # Handle automatic norm insertion
        if '{norm}' in dataset_type:
            dataset_type = dataset_type.format(norm=self.norm)
        elif 'adv_' in dataset_type and f'_{self.norm}' not in dataset_type:
            dataset_type = f"{dataset_type}_{self.norm}"
        
        filename = f"{dataset_type}.pickle"
        
        if filename not in self.available_files:
            available = list(self.available_files.keys())
            raise ValueError(f"Dataset '{dataset_type}' not found. Available: {available}")
        
        filepath = self.available_files[filename]['path']
        
        print(f"Loading {dataset_type} from {filepath}")
        with open(filepath, 'rb') as f:
            data, labels = pickle.load(f)
        
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Unique labels: {len(np.unique(labels))}")
        
        return data, labels
    
    def load_universal_perturbation(self):
        """Load the universal perturbation"""
        filename = f"universal_perturbation_{self.norm}.pickle"
        
        if filename not in self.available_files:
            available = list(self.available_files.keys())
            raise ValueError(f"Universal perturbation not found. Available: {available}")
        
        filepath = self.available_files[filename]['path']
        
        print(f"Loading universal perturbation from {filepath}")
        with open(filepath, 'rb') as f:
            perturbation = pickle.load(f)
        
        print(f"  Perturbation shape: {perturbation.shape}")
        print(f"  Perturbation range: [{perturbation.min():.6f}, {perturbation.max():.6f}]")
        print(f"  Perturbation norm: {torch.norm(perturbation).item():.6f}")
        
        return perturbation
    
    def get_dataloader(self, dataset_type, batch_size=128, shuffle=True, 
                      num_workers=4, transform=None):
        """
        Get a PyTorch DataLoader for the specified dataset
        
        Args:
            dataset_type: Type of dataset to load
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            transform: Optional transform to apply
        
        Returns:
            DataLoader: PyTorch DataLoader object
        """
        data, labels = self.load_dataset(dataset_type)
        dataset = PickleDataset(data, labels, transform=transform)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader
    
    def get_mixed_dataloader(self, dataset_types, ratios=None, batch_size=128, 
                           shuffle=True, num_workers=4, max_samples_per_type=None):
        """
        Create a mixed DataLoader with multiple dataset types
        
        Args:
            dataset_types: List of dataset types to mix
            ratios: List of ratios for each dataset type (must sum to 1.0)
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            max_samples_per_type: Maximum samples to take from each dataset
        
        Returns:
            DataLoader: Mixed DataLoader
        """
        if ratios is None:
            ratios = [1.0 / len(dataset_types)] * len(dataset_types)
        
        if len(ratios) != len(dataset_types):
            raise ValueError("Number of ratios must match number of dataset types")
        
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Load all datasets
        all_data = []
        all_labels = []
        
        for dataset_type, ratio in zip(dataset_types, ratios):
            data, labels = self.load_dataset(dataset_type)
            
            # Limit samples if specified
            if max_samples_per_type is not None:
                n_samples = min(len(data), max_samples_per_type)
                indices = np.random.choice(len(data), n_samples, replace=False)
                data = data[indices]
                labels = labels[indices]
            
            # Calculate how many samples to take based on ratio
            n_samples = int(len(data) * ratio)
            if n_samples > 0:
                indices = np.random.choice(len(data), n_samples, replace=False)
                all_data.append(data[indices])
                all_labels.append(labels[indices])
        
        # Combine all datasets
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Mixed dataset: {len(combined_data)} samples")
        for i, (dataset_type, ratio) in enumerate(zip(dataset_types, ratios)):
            n_samples = len(all_data[i])
            print(f"  {dataset_type}: {n_samples} samples ({ratio*100:.1f}%)")
        
        # Create dataset and dataloader
        dataset = PickleDataset(combined_data, combined_labels)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader
    
    def get_adversarial_training_loader(self, adv_type='mixed', clean_ratio=0.5, 
                                      batch_size=128, shuffle=True):
        """
        Get a DataLoader specifically designed for adversarial training
        
        Args:
            adv_type: Type of adversarial training
                - 'persample': Only per-sample adversarial + clean
                - 'universal': Only universal adversarial + clean  
                - 'mixed': Mix of per-sample, universal, and clean
            clean_ratio: Ratio of clean samples
            batch_size: Batch size
            shuffle: Whether to shuffle
        
        Returns:
            DataLoader: Adversarial training DataLoader
        """
        if adv_type == 'persample':
            dataset_types = ['clean_train', f'adv_persample_train_{self.norm}']
            ratios = [clean_ratio, 1.0 - clean_ratio]
            
        elif adv_type == 'universal':
            dataset_types = ['clean_train', f'adv_universal_train_{self.norm}']
            ratios = [clean_ratio, 1.0 - clean_ratio]
            
        elif adv_type == 'mixed':
            # Mix all three types
            adv_ratio = 1.0 - clean_ratio
            dataset_types = ['clean_train', 
                           f'adv_persample_train_{self.norm}',
                           f'adv_universal_train_{self.norm}']
            ratios = [clean_ratio, adv_ratio/2, adv_ratio/2]
            
        else:
            raise ValueError(f"Unknown adv_type: {adv_type}")
        
        print(f"\nCreating adversarial training loader ({adv_type}):")
        return self.get_mixed_dataloader(dataset_types, ratios, batch_size, shuffle)

def example_usage():
    """Comprehensive example of how to use the CompleteDatasetManager"""
    
    # Initialize dataset manager
    manager = CompleteDatasetManager('cifar10', 'resnet56', 'sdn_ic_only', norm='l2')
    
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic dataset loading")
    print("="*50)
    
    # Load clean validation data
    clean_valid_loader = manager.get_dataloader('clean_valid', batch_size=128)
    
    # Load per-sample adversarial training data
    persample_train_loader = manager.get_dataloader('adv_persample_train', batch_size=128)
    
    # Load universal adversarial validation data
    universal_valid_loader = manager.get_dataloader('adv_universal_valid', batch_size=128)
    
    print("\n" + "="*50)
    print("EXAMPLE 2: Mixed adversarial training")
    print("="*50)
    
    # Mixed adversarial training (clean + per-sample + universal)
    mixed_train_loader = manager.get_adversarial_training_loader(
        adv_type='mixed', clean_ratio=0.4, batch_size=128
    )
    
    # Per-sample adversarial training only
    persample_train_loader = manager.get_adversarial_training_loader(
        adv_type='persample', clean_ratio=0.6, batch_size=128
    )
    
    print("\n" + "="*50)
    print("EXAMPLE 3: Custom mixing")
    print("="*50)
    
    # Custom mix: 40% clean, 30% per-sample adv, 30% universal adv
    custom_loader = manager.get_mixed_dataloader(
        dataset_types=['clean_train', 'adv_persample_train', 'adv_universal_train'],
        ratios=[0.4, 0.3, 0.3],
        batch_size=128
    )
    
    print("\n" + "="*50)
    print("EXAMPLE 4: Universal perturbation")
    print("="*50)
    
    # Load the universal perturbation itself
    universal_perturbation = manager.load_universal_perturbation()
    
    # Example training loop snippet
    print(f"\nExample training loop with mixed adversarial data:")
    for batch_idx, (data, target) in enumerate(mixed_train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        if batch_idx >= 2:  # Just show first few batches
            break

if __name__ == "__main__":
    example_usage()