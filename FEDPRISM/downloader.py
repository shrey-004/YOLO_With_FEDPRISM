import ssl
import os
from torchvision import datasets

# Disable SSL verification to avoid certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

def download_all():
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Downloading datasets to {os.path.abspath(data_dir)}...")

    # CIFAR-10
    print("\n[1/4] Downloading CIFAR-10...")
    try:
        datasets.CIFAR10(root=data_dir, train=True, download=True)
        datasets.CIFAR10(root=data_dir, train=False, download=True)
        print("CIFAR-10 Downloaded.")
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")

    # CIFAR-100
    print("\n[2/4] Downloading CIFAR-100...")
    try:
        datasets.CIFAR100(root=data_dir, train=True, download=True)
        datasets.CIFAR100(root=data_dir, train=False, download=True)
        print("CIFAR-100 Downloaded.")
    except Exception as e:
        print(f"Error downloading CIFAR-100: {e}")

    # FashionMNIST
    print("\n[3/4] Downloading FashionMNIST...")
    try:
        datasets.FashionMNIST(root=data_dir, train=True, download=True)
        datasets.FashionMNIST(root=data_dir, train=False, download=True)
        print("FashionMNIST Downloaded.")
    except Exception as e:
        print(f"Error downloading FashionMNIST: {e}")

    # SVHN
    print("\n[4/4] Downloading SVHN...")
    try:
        datasets.SVHN(root=data_dir, split='train', download=True)
        datasets.SVHN(root=data_dir, split='test', download=True)
        print("SVHN Downloaded.")
    except Exception as e:
        print(f"Error downloading SVHN: {e}")

    print("\nAll downloads attempted.")

if __name__ == "__main__":
    download_all()
