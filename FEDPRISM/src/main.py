import argparse
import os
import torch
import numpy as np
import pandas as pd
from src.data.datasets import get_dataset
from src.data.partition import partition_data
from src.models.lenet import LeNet5
from src.algorithms.fedavg import FedAvgServer
from src.algorithms.local import LocalServer
from src.algorithms.fedclust import FedClustServer
from src.algorithms.fedprism import FedPrismServer

import yaml

def args_parser():
    parser = argparse.ArgumentParser()
    # Load config first
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Define args with defaults from config
    parser.add_argument('--epochs', type=int, default=config.get('epochs', 100), help="rounds of training")
    parser.add_argument('--num_users', type=int, default=config.get('num_users', 100), help="number of users: K")
    parser.add_argument('--frac', type=float, default=config.get('frac', 0.1), help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=config.get('local_ep', 10), help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=config.get('local_bs', 32), help="local batch size: B")
    parser.add_argument('--bs', type=int, default=config.get('bs', 32), help="test batch size")
    parser.add_argument('--lr', type=float, default=config.get('lr', 0.01), help="learning rate")
    parser.add_argument('--momentum', type=float, default=config.get('momentum', 0.9), help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # Model arguments
    parser.add_argument('--model', type=str, default='lenet', help='model name')
    
    # Other arguments
    parser.add_argument('--dataset', type=str, default=config.get('dataset', 'cifar10'), help="name of dataset")
    parser.add_argument('--partition', type=str, default=config.get('partition', 'dirichlet'), help="partition type")
    parser.add_argument('--alpha', type=float, default=config.get('alpha', 0.5), help="alpha for dirichlet")
    parser.add_argument('--algorithm', type=str, default=config.get('algorithm', 'fedavg'), help='algorithm name')
    parser.add_argument('--gpu', type=int, default=config.get('gpu', 0), help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=config.get('seed', 1), help='random seed (default: 1)')
    
    # Clustering arguments
    parser.add_argument('--num_clusters', type=int, default=config.get('num_clusters', 5), help="number of clusters")
    parser.add_argument('--clustering_method', type=str, default=config.get('clustering_method', 'kmeans'), help="clustering method")
    parser.add_argument('--alpha_coef', type=float, default=config.get('alpha_coef', 0.5), help="ensemble coefficient for FedPRISM")
    parser.add_argument('--clustering_freq', type=int, default=config.get('clustering_freq', 10), help="clustering frequency")
    parser.add_argument('--num_assignments', type=int, default=config.get('num_assignments', 2), help="number of soft assignments (m)")
    parser.add_argument('--trainable_alpha', type=lambda x: (str(x).lower() == 'true'), default=config.get('trainable_alpha', False), help="trainable alpha")

    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    train_dataset, test_dataset = get_dataset(args.dataset, './data')
    
    # Partition data
    dict_users_train, dict_users_test = partition_data(train_dataset, test_dataset, args.num_users, args.partition, args.alpha)
    
    # Initialize Model
    if args.dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
        
    img_size = train_dataset[0][0].shape
    
    net_glob = LeNet5(input_channels=img_size[0], num_classes=num_classes)
    
    # Initialize Server
    if args.algorithm == 'fedavg':
        server = FedAvgServer(args, train_dataset, dict_users_train, net_glob, test_dataset, dict_users_test)
    elif args.algorithm == 'local':
        server = LocalServer(args, train_dataset, dict_users_train, net_glob, test_dataset, dict_users_test)
    elif args.algorithm == 'fedclust':
        server = FedClustServer(args, train_dataset, dict_users_train, net_glob, test_dataset, dict_users_test)
    elif args.algorithm == 'fedprism':
        server = FedPrismServer(args, train_dataset, dict_users_train, net_glob, test_dataset, dict_users_test)
    else:
        raise ValueError(f"Algorithm {args.algorithm} not implemented")
        
    print(f"Starting training for {args.algorithm} on {args.dataset} with {args.partition} (alpha={args.alpha})")
    
    # Train
    if args.algorithm == 'local':
        models, loss_train, acc_test, acc_test_local = server.train()
        alpha_history = []
    elif args.algorithm == 'fedprism':
        model, loss_train, alpha_history, acc_test, acc_test_local = server.train()
    else:
        model, loss_train, acc_test, acc_test_local = server.train()
        alpha_history = []
        
    # Save results
    save_dir = './results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"{args.algorithm}_{args.dataset}_{args.partition}_{args.alpha}_{args.clustering_method}.csv"
    
    # Pad alpha_history if needed
    if len(alpha_history) < len(loss_train):
        alpha_history = [0.0] * len(loss_train)
        
    df = pd.DataFrame({
        'round': range(len(loss_train)), 
        'loss': loss_train, 
        'accuracy': acc_test,
        'local_accuracy': acc_test_local,
        'avg_alpha': alpha_history
    })
    df.to_csv(os.path.join(save_dir, filename), index=False)
    
    print(f"Training finished. Results saved to {filename}")
    
    # Note: Plotting is now done separately or at the end of all experiments
    # You can run: python -m src.utils.plotting to generate all plots

if __name__ == '__main__':
    main()
