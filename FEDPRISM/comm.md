# Federated Learning Benchmark

This repository contains implementations of FedAvg, FedClust, and Fed-PRISM (with variants) for benchmarking on CIFAR-10, CIFAR-100, SVHN, and FMNIST.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

### Quick Verification (Dry Run)
Runs a minimal set of experiments (1 epoch, FMNIST) to verify code works.
```bash
python run_experiments.py --mode dry_run
```

### Full Benchmark
Runs the full suite of experiments (WARNING: Takes a long time).
```bash
python run_experiments.py --mode full
```

### Custom Run
You can run `src/main.py` directly for specific configurations:
```bash
python src/main.py --dataset cifar10 --algorithm fedprism --clustering_method covariance --alpha 0.1 --epochs 100
```

## Results
Results (CSV logs) are saved in `results/`.
Plots can be generated using:
```bash
python src/utils/plotting.py
```
