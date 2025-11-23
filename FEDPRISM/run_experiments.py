import os
import subprocess
import argparse
import itertools

def run_command(cmd, dry_run=False):
    print(f"Running: {cmd}")
    if not dry_run:
        subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    parser.add_argument('--mode', type=str, default='dry_run', choices=['dry_run', 'full'], help="Mode: dry_run (fast check) or full (all experiments)")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    # Configuration Grid
    datasets = ['cifar10', 'cifar100', 'svhn', 'fmnist']
    alphas = [0.5, 0.3, 0.1]
    
    # Algorithms and their specific params
    # (algo_name, clustering_method, alpha_coef, trainable_alpha)
    algos = []
    
    # FedAvg
    algos.append(('fedavg', 'kmeans', 0.5, False)) 
    
    # FedClust
    algos.append(('fedclust', 'kmeans', 0.5, False))
    
    # Fed-PRISM variants (Fixed Alpha)
    algos.append(('fedprism', 'kmeans', 0.5, False))
    algos.append(('fedprism', 'single', 0.5, False))
    algos.append(('fedprism', 'average', 0.5, False))
    algos.append(('fedprism', 'covariance', 0.5, False))
    
    # Fed-PRISM variants (Trainable Alpha)
    algos.append(('fedprism', 'kmeans', 0.5, True))
    algos.append(('fedprism', 'covariance', 0.5, True)) 
    # Note: Adding trainable alpha for all variants might be too many runs, 
    # but user asked for "all kinds of results". 
    # I'll add the most important ones (KMeans and Covariance) to save some time, 
    # or just add all if user insists on "all permutations".
    # User said: "all permuations and combiantios". Okay, adding all.
    algos.append(('fedprism', 'single', 0.5, True))
    algos.append(('fedprism', 'average', 0.5, True))

    if args.mode == 'dry_run':
        print("=== DRY RUN MODE ===")
        # Run 1 epoch, 1 dataset, 1 alpha, subset of algos
        epochs = 1
        datasets = ['fmnist'] 
        alphas = [0.5]
        num_users = 10
        frac = 0.1
        # Reduce algos for dry run
        algos = [('fedprism', 'kmeans', 0.5, True)]
    else:
        print("=== FULL EXPERIMENT MODE ===")
        epochs = 100 
        num_users = 100
        frac = 0.1

    # Iterate
    for dataset in datasets:
        for alpha in alphas:
            for algo, cluster_method, coef, train_alpha in algos:
                cmd = (f"python -m src.main --dataset {dataset} --algorithm {algo} "
                       f"--partition dirichlet --alpha {alpha} "
                       f"--epochs {epochs} --num_users {num_users} --frac {frac} "
                       f"--gpu {args.gpu} --clustering_method {cluster_method} --alpha_coef {coef} "
                       f"--trainable_alpha {train_alpha}")
                
                try:
                    run_command(cmd, dry_run=False) 
                except subprocess.CalledProcessError as e:
                    print(f"Error running command: {cmd}")
                    print(e)

    print("All experiments finished.")
    
    # Generate organized plots grouped by dataset and alpha
    print("\nGenerating plots...")
    from src.utils.plotting import plot_results
    plot_results('./results')
    print("\nPlots generated! Check ./results/ subdirectories for organized visualizations.")

if __name__ == '__main__':
    main()
