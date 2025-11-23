import ssl
# Disable SSL verification to avoid certificate errors (common with CIFAR-10 download)
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torchvision import datasets, transforms
import os

def get_dataset(dataset_name, data_dir='./data'):
    """
    Returns train and test datasets for the specified dataset.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
        
    return train_dataset, test_dataset
