import os
import subprocess

def run_experiment(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    datasets = ['cifar10', 'cifar100', 'svhn', 'fmnist']
    alphas = [0.5, 0.3, 0.1]
    algorithms = ['fedavg', 'fedclust', 'fedprism']
    clustering_methods = ['kmeans', 'single', 'average', 'covariance']
    
    # Reduced set for testing/demo purposes, user can expand
    # User asked for: "multiple runs of all algorthms benchmarked across these datst"
    # "make it highly non iid using dirchilet 0.5 ,0.3 and 0.1"
    
    # WARNING: This full grid is HUGE. 
    # 4 datasets * 3 alphas * (1 fedavg + 1 fedclust + 4 fedprism) = 4 * 3 * 6 = 72 runs.
    # Each run 100-300 rounds. This will take DAYS.
    
    # I will create a 'demo' run script that runs 1 epoch of each to verify.
    # And a 'full' run script.
    
    pass

if __name__ == '__main__':
    # This file is just a placeholder for the logic I'll put in a python script or bat file
    pass
