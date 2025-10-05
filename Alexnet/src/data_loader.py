"""Data loading and preprocessing utilities."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def get_transforms(mean, std, train=True):
    """
    Get data transformations for training or validation/test.
    
    Args:
        mean (list): Channel-wise mean values
        std (list): Channel-wise standard deviation values
        train (bool): Whether to apply training augmentations
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    return transform


def get_data_loaders(config):
    """
    Create train, validation, and test data loaders.
    
    Args:
        config (dict): Configuration dictionary containing dataset parameters
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    data_dir = config['dataset']['data_dir']
    batch_size = config['training']['batch_size']
    valid_size = config['training']['valid_size']
    shuffle = config['training']['shuffle']
    random_seed = config['training']['random_seed']
    num_workers = config['dataset']['num_workers']
    pin_memory = config['dataset']['pin_memory']
    mean = config['dataset']['mean']
    std = config['dataset']['std']
    
    # Get transforms
    train_transform = get_transforms(mean, std, train=True)
    test_transform = get_transforms(mean, std, train=False)
    
    # Load training and validation datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    valid_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create train/validation split
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, valid_loader, test_loader


def get_class_names():
    """
    Get CIFAR-10 class names.
    
    Returns:
        list: List of class names
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

