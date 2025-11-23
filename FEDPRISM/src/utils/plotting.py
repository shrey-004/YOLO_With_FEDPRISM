import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from collections import defaultdict

def plot_results(results_dir='./results'):
    """
    Generate organized plots grouped by dataset and alpha value.
    Creates separate subdirectories for each dataset-alpha combination.
    """
    files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not files:
        print("No CSV files found in results directory.")
        return
    
    # Group files by dataset and alpha
    # Filename format: {algorithm}_{dataset}_{partition}_{alpha}_{clustering_method}.csv
    grouped_files = defaultdict(list)
    
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace('.csv', '').split('_')
        
        if len(parts) >= 4:
            dataset = parts[1]
            alpha = parts[3]
            key = f"{dataset}_alpha{alpha}"
            grouped_files[key].append(f)
    
    # Create plots for each group
    for group_key, group_files in grouped_files.items():
        # Create subdirectory for this group
        group_dir = os.path.join(results_dir, group_key)
        os.makedirs(group_dir, exist_ok=True)
        
        print(f"\nGenerating plots for {group_key}...")
        
        # Plot Loss
        plt.figure(figsize=(12, 7))
        for f in group_files:
            df = pd.read_csv(f)
            label = os.path.basename(f).replace('.csv', '').replace(f"{group_key.split('_alpha')[0]}_dirichlet_{group_key.split('_alpha')[1]}_", '')
            plt.plot(df['round'], df['loss'], label=label, linewidth=2)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training Loss - {group_key.replace("_", " ").title()}', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = os.path.join(group_dir, 'loss_comparison.png')
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Loss plot: {loss_path}")
        
        # Plot Alpha (if available)
        has_alpha = False
        plt.figure(figsize=(12, 7))
        for f in group_files:
            df = pd.read_csv(f)
            if 'avg_alpha' in df.columns and df['avg_alpha'].sum() > 0:
                label = os.path.basename(f).replace('.csv', '').replace(f"{group_key.split('_alpha')[0]}_dirichlet_{group_key.split('_alpha')[1]}_", '')
                plt.plot(df['round'], df['avg_alpha'], label=label, linewidth=2)
                has_alpha = True
                
        if has_alpha:
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Average Alpha', fontsize=12)
            plt.title(f'Alpha Evolution - {group_key.replace("_", " ").title()}', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            alpha_path = os.path.join(group_dir, 'alpha_evolution.png')
            plt.savefig(alpha_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Alpha plot: {alpha_path}")
        else:
            plt.close()
    
        # Plot Global Accuracy
        plt.figure(figsize=(12, 7))
        for f in group_files:
            df = pd.read_csv(f)
            if 'accuracy' in df.columns:
                label = os.path.basename(f).replace('.csv', '').replace(f"{group_key.split('_alpha')[0]}_dirichlet_{group_key.split('_alpha')[1]}_", '')
                plt.plot(df['round'], df['accuracy'], label=label, linewidth=2)
                
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Global Test Accuracy (%)', fontsize=12)
        plt.title(f'Global Test Accuracy - {group_key.replace("_", " ").title()}', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        acc_path = os.path.join(group_dir, 'accuracy_comparison.png')
        plt.savefig(acc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Global Accuracy plot: {acc_path}")
    
        # Plot Local Accuracy
        has_local = False
        plt.figure(figsize=(12, 7))
        for f in group_files:
            df = pd.read_csv(f)
            if 'local_accuracy' in df.columns:
                label = os.path.basename(f).replace('.csv', '').replace(f"{group_key.split('_alpha')[0]}_dirichlet_{group_key.split('_alpha')[1]}_", '')
                plt.plot(df['round'], df['local_accuracy'], label=label, linewidth=2)
                has_local = True
                
        if has_local:
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Avg Local Test Accuracy (%)', fontsize=12)
            plt.title(f'Local Test Accuracy - {group_key.replace("_", " ").title()}', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            local_path = os.path.join(group_dir, 'local_accuracy_comparison.png')
            plt.savefig(local_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Local Accuracy plot: {local_path}")
        else:
            plt.close()
    
    print(f"\nAll plots generated! Check subdirectories in {results_dir}/")

if __name__ == '__main__':
    plot_results()
