================================================================================
FEDERATED LEARNING EXPERIMENTS - DETAILED DOCUMENTATION
================================================================================

This document provides a comprehensive overview of the federated learning 
experiments conducted in this project, focusing specifically on CIFAR-10 and 
Fashion-MNIST (FMNIST) datasets.

================================================================================
1. OVERVIEW
================================================================================

This project implements and benchmarks multiple federated learning algorithms 
under non-IID (non-Independent and Identically Distributed) data settings. 
The primary goal is to evaluate the performance of Fed-PRISM (Personalized 
Federated Learning with Adaptive Clustering) against baseline algorithms 
including FedAvg, Local Training, and FedClust.

Key Research Questions:
- How do different federated learning algorithms perform under varying degrees 
  of data heterogeneity (non-IID settings)?
- Does adaptive alpha (personalization coefficient) improve performance 
  compared to fixed alpha in Fed-PRISM?
- Which clustering methods are most effective for identifying client similarity 
  in heterogeneous federated settings?

================================================================================
2. DATASETS
================================================================================

2.1 CIFAR-10
-------------
- Description: Color image classification dataset
- Number of Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, 
  horse, ship, truck)
- Image Dimensions: 32x32 pixels, 3 channels (RGB)
- Training Samples: 50,000 images
- Test Samples: 10,000 images
- Use Case: Evaluates algorithm performance on complex, multi-channel image 
  data with fine-grained visual distinctions

2.2 Fashion-MNIST (FMNIST)
---------------------------
- Description: Grayscale fashion product image classification dataset
- Number of Classes: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, 
  Shirt, Sneaker, Bag, Ankle boot)
- Image Dimensions: 28x28 pixels, 1 channel (grayscale)
- Training Samples: 60,000 images
- Test Samples: 10,000 images
- Use Case: Provides a simpler baseline for evaluating algorithms on grayscale 
  data with clear categorical distinctions

================================================================================
3. EXPERIMENTAL SETUP
================================================================================

3.1 Federated Learning Configuration
-------------------------------------
- Number of Clients: 100
- Client Participation Rate: 10% per round (10 clients selected randomly)
- Total Communication Rounds: 100
- Local Training Epochs per Round: 10
- Local Batch Size: 32
- Test Batch Size: 32
- Optimizer: SGD with momentum
- Learning Rate: 0.01
- Momentum: 0.9
- Random Seed: 1 (for reproducibility)

3.2 Data Partitioning (Non-IID Settings)
-----------------------------------------
The experiments use Dirichlet distribution-based partitioning to create 
realistic non-IID data distributions across clients. The Dirichlet parameter 
(alpha) controls the degree of heterogeneity:

Alpha Values Tested:
- α = 0.5: Moderate heterogeneity
- α = 0.3: High heterogeneity
- α = 0.1: Extreme heterogeneity (highly non-IID)

Lower alpha values result in more skewed class distributions across clients, 
simulating real-world scenarios where different clients have vastly different 
data distributions (e.g., hospitals with different patient demographics, 
mobile devices with different user behaviors).

3.3 Model Architecture
----------------------
- Model: LeNet-5 (Convolutional Neural Network)
- Architecture:
  * Convolutional Layer 1: 6 filters, 5x5 kernel
  * Max Pooling: 2x2
  * Convolutional Layer 2: 16 filters, 5x5 kernel
  * Max Pooling: 2x2
  * Fully Connected Layer 1: 120 units
  * Fully Connected Layer 2: 84 units
  * Output Layer: 10 units (for both CIFAR-10 and FMNIST)
- Activation: ReLU (Rectified Linear Unit)
- Input Channels: 3 for CIFAR-10, 1 for FMNIST

================================================================================
4. ALGORITHMS EVALUATED
================================================================================

4.1 FedAvg (Federated Averaging)
---------------------------------
- Description: Baseline federated learning algorithm that aggregates client 
  model updates by simple averaging
- Key Characteristics:
  * No personalization
  * All clients receive identical global model
  * Simple and communication-efficient
- Expected Performance: Degrades under high non-IID settings

4.2 Local Training
------------------
- Description: Each client trains independently without any aggregation
- Key Characteristics:
  * Maximum personalization
  * No knowledge sharing between clients
  * Serves as lower bound for collaborative learning
- Expected Performance: Poor on clients with limited data, but may excel 
  on clients with sufficient local data

4.3 FedClust (Federated Clustering)
------------------------------------
- Description: Clusters clients based on model updates and maintains separate 
  models per cluster
- Key Characteristics:
  * Hard cluster assignment (each client belongs to one cluster)
  * Number of Clusters: 5
  * Clustering Frequency: Every 10 rounds
  * Uses K-Means clustering on flattened model weights
- Expected Performance: Better than FedAvg under non-IID, but limited by 
  hard assignments

4.4 Fed-PRISM (Federated Personalized Learning with Soft Clustering)
---------------------------------------------------------------------
- Description: Advanced personalized federated learning with soft cluster 
  assignments and adaptive ensemble coefficients
- Key Characteristics:
  * Soft cluster assignment (clients can belong to multiple clusters)
  * Number of Clusters (K): 5
  * Number of Soft Assignments (m): 2 (each client assigned to top-2 clusters)
  * Clustering Frequency: Every 10 rounds
  * Personalized model = α × Global Model + (1-α) × Weighted Cluster Models

Fed-PRISM Variants Tested:

A. Clustering Methods (with Fixed Alpha = 0.5):
   1. K-Means: Standard centroid-based clustering on model weights
   2. Single Linkage: Hierarchical clustering with minimum distance
   3. Average Linkage: Hierarchical clustering with average distance
   4. Covariance: Spectral clustering using cosine similarity on model 
      weight gradients (updates from personalized starting point)

B. Adaptive Alpha Variants:
   For each clustering method above, we also test with trainable alpha:
   - Initial Alpha: 0.5
   - Alpha Update Rule: 
     * If Global Model Loss < Cluster Model Loss: α = min(1.0, α + 0.05)
     * Else: α = max(0.0, α - 0.05)
   - Rationale: Dynamically adjusts personalization based on which model 
     (global vs. cluster) performs better for each client

Total Fed-PRISM Configurations: 8
- 4 clustering methods × 2 alpha settings (fixed vs. adaptive)

================================================================================
5. EVALUATION METRICS
================================================================================

5.1 Training Loss
-----------------
- Average training loss across selected clients per communication round
- Indicates convergence behavior and training efficiency

5.2 Global Test Accuracy
-------------------------
- Accuracy of the global model on the centralized test set
- Measures generalization to unseen data using the shared global model
- Relevant for FedAvg and as a baseline for other algorithms

5.3 Local Test Accuracy (Personalized)
---------------------------------------
- Average accuracy of personalized models on local client test sets
- Each client's test set mirrors the distribution of their training data
- Primary metric for evaluating personalization effectiveness
- Most relevant for Fed-PRISM, FedClust, and Local Training

5.4 Alpha Evolution (Fed-PRISM only)
-------------------------------------
- Tracks the average alpha coefficient across all clients over rounds
- Only applicable for adaptive alpha variants
- Provides insights into how personalization adapts over training

================================================================================
6. EXPERIMENTAL MATRIX
================================================================================

For CIFAR-10 and Fashion-MNIST, the following experiments are conducted:

Datasets: 2 (CIFAR-10, FMNIST)
Alpha Values (Non-IID levels): 3 (0.5, 0.3, 0.1)
Algorithms:
  - FedAvg: 1 configuration
  - Local: 1 configuration
  - FedClust: 1 configuration
  - Fed-PRISM: 8 configurations (4 clustering methods × 2 alpha settings)

Total Experiments per Dataset: 11 algorithm configurations × 3 alpha values = 33
Total Experiments (Both Datasets): 33 × 2 = 66 experiments

Each experiment runs for 100 communication rounds with consistent hyperparameters.

================================================================================
7. EXPECTED OUTCOMES & HYPOTHESES
================================================================================

7.1 Impact of Non-IID Severity (Alpha)
---------------------------------------
Hypothesis: As alpha decreases (more non-IID):
- FedAvg performance degrades significantly
- Local training becomes more competitive
- Fed-PRISM's advantage over baselines increases
- Adaptive alpha shows greater benefits

7.2 Clustering Method Comparison
---------------------------------
Hypothesis:
- Covariance-based clustering (using gradients) captures client similarity 
  better than weight-based methods
- K-Means provides good baseline performance
- Hierarchical methods (single/average) may struggle with high-dimensional 
  weight spaces

7.3 Adaptive vs. Fixed Alpha
-----------------------------
Hypothesis:
- Adaptive alpha improves performance, especially under extreme non-IID (α=0.1)
- Clients with poor cluster matches will increase alpha (rely more on global)
- Clients with good cluster matches will decrease alpha (rely more on clusters)
- Alpha evolution stabilizes after initial rounds

7.4 Dataset-Specific Observations
----------------------------------
CIFAR-10:
- More challenging due to color channels and visual complexity
- Larger performance gaps between algorithms expected
- Clustering methods may show more pronounced differences

Fashion-MNIST:
- Simpler grayscale images may converge faster
- All algorithms expected to achieve higher absolute accuracy
- Relative performance trends should mirror CIFAR-10

================================================================================
8. RESULTS STORAGE & ANALYSIS
================================================================================

8.1 Results Directory Structure
--------------------------------
All experimental results are saved in: ./results/

File Naming Convention:
{algorithm}_{dataset}_{partition}_{alpha}_{clustering_method}.csv

Examples:
- fedavg_cifar10_dirichlet_0.5_kmeans.csv
- fedprism_fmnist_dirichlet_0.1_covariance.csv
- local_cifar10_dirichlet_0.3_kmeans.csv

8.2 CSV File Contents
----------------------
Each CSV file contains the following columns:
- round: Communication round number (0 to 99)
- loss: Average training loss for that round
- accuracy: Global test accuracy (%)
- local_accuracy: Average local personalized test accuracy (%)
- avg_alpha: Average alpha across clients (0.0 for non-Fed-PRISM algorithms)

8.3 Visualization
-----------------
Plots are generated using: python -m src.utils.plotting

Generated plots include:
- Accuracy vs. Communication Rounds (comparing all algorithms)
- Loss vs. Communication Rounds
- Local vs. Global Accuracy Comparison
- Alpha Evolution over Rounds (for adaptive Fed-PRISM variants)
- Grouped comparisons by dataset and alpha value

================================================================================
9. REPRODUCIBILITY
================================================================================

9.1 Running Experiments
------------------------
Full Benchmark (All 66 Experiments):
  python run_experiments.py --mode full --gpu 0

Quick Verification (Dry Run):
  python run_experiments.py --mode dry_run

Single Experiment Example:
  python -m src.main --dataset cifar10 --algorithm fedprism \
    --clustering_method covariance --alpha 0.1 --epochs 100 \
    --trainable_alpha True

9.2 Hardware Requirements
--------------------------
- GPU: Recommended for faster training (CUDA-compatible)
- CPU: Supported but significantly slower
- RAM: Minimum 8GB recommended
- Storage: ~500MB for datasets and results

9.3 Software Dependencies
--------------------------
- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- scikit-learn
- scipy
- PyYAML

Install via: pip install -r requirements.txt

9.4 Random Seed
---------------
All experiments use seed=1 for reproducibility. This ensures:
- Consistent data partitioning across runs
- Identical client selection sequences
- Reproducible model initialization

================================================================================
10. KEY INSIGHTS FOR REPORT
================================================================================

When writing your report, consider highlighting:

1. Non-IID Challenge: Emphasize how real-world federated learning faces 
   heterogeneous data distributions, making simple averaging (FedAvg) 
   insufficient.

2. Personalization Trade-off: Fed-PRISM balances global knowledge (via global 
   model) and local specialization (via cluster models) through the alpha 
   coefficient.

3. Adaptive Learning: The adaptive alpha mechanism allows Fed-PRISM to 
   automatically adjust personalization per client, removing the need for 
   manual hyperparameter tuning.

4. Soft Clustering Advantage: Unlike FedClust's hard assignments, Fed-PRISM's 
   soft clustering allows clients to benefit from multiple relevant clusters, 
   improving robustness.

5. Scalability: The experimental setup with 100 clients and varying non-IID 
   levels demonstrates scalability and robustness across diverse scenarios.

6. Comprehensive Evaluation: Testing on both CIFAR-10 (complex, color) and 
   Fashion-MNIST (simple, grayscale) validates algorithm generalizability.

================================================================================
11. LIMITATIONS & FUTURE WORK
================================================================================

Current Limitations:
- Fixed number of clusters (K=5) - could be optimized per dataset
- Limited to image classification tasks
- Assumes all clients have sufficient data for local training
- Communication costs not explicitly measured

Potential Extensions:
- Dynamic cluster number selection
- Privacy-preserving clustering methods
- Communication-efficient aggregation strategies
- Extension to other domains (NLP, time-series)
- Handling client dropout and asynchronous updates

================================================================================
END OF DOCUMENT
================================================================================

For questions or clarifications, refer to:
- config.yaml: Hyperparameter settings
- src/algorithms/: Algorithm implementations
- run_experiments.py: Experiment orchestration script
- README.md: Quick start guide

Last Updated: 2025-11-23
