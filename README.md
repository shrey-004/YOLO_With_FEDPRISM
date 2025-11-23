Here is a clean, well-structured **README.md** for your project, fully based on the content of your report .

You can copy-paste directly into GitHub.

---

# 🚗 FedPrism — Federated Personalized Object Detection for IoV

### *YOLOv12 + Dynamic Soft-Clustering for Non-IID FL*

This repository contains the implementation of **FedPrism (Federated Personalized Relevance-based Intelligent Soft-assignment Model)** — a new federated learning framework designed for **Internet of Vehicles (IoV)** and extreme **non-IID object detection** scenarios.
It also includes a **comparative YOLO benchmarking study** (YOLOv5 vs YOLOv8 vs YOLOv12) under federated settings.

---

## 📌 Key Features

### ✅ **1. YOLO Architecture Benchmarking**

We compare three YOLO architectures in a federated setup:

| Model       | Final Loss | Improvement vs YOLOv5 |
| ----------- | ---------- | --------------------- |
| YOLOv5      | 0.1000     | –                     |
| YOLOv8      | 0.0702     | 29.8%                 |
| **YOLOv12** | **0.0617** | **38.3%**             |

➡️ **YOLOv12 is the best FL baseline** for IoV due to modern design, C2f blocks, and Xavier initialization optimized for non-IID training.

---

### ✅ **2. FedPrism Algorithm**

FedPrism introduces **dynamic clustering + soft assignment + adaptive blending**, overcoming limitations of:

* **FedAvg** (single global model → poor with non-IID)
* **FedClust** (static clustering, cannot adapt to drift)
* **IFCA** (hard assignment; only 1 cluster per client)

#### ✔ Core ideas:

* Maintain **1 global model ΘG**
* Maintain **K cluster models Θ₁…ΘK**
* Each client gets a **personalized mix**:

[
\Theta_{i,pers} = \alpha \Theta_G + (1-\alpha)\sum_{j\in W_i} W_{i,j}\Theta_j
]

Where:

* **α** dynamically learns how much global vs cluster knowledge to use
* **Wᵢⱼ** are *soft-assignment weights* computed using cosine similarity + softmax
* Re-clustering happens **every C rounds**
* Clients may belong to **multiple clusters (Top-M assignment)**

---

## 🚀 Performance Highlights

### 🔹 **Breakthrough Result (Extreme Non-IID—CIFAR-100 Pathological Split)**

| Method              | Accuracy   |
| ------------------- | ---------- |
| FedClust            | 17–20%     |
| **FedPrism (ours)** | **58–62%** |

➡️ **FedPrism achieves 3× better accuracy** than FedClust on pathological non-IID settings. 

---

## 📁 Repository Structure

```
FedPrism-IoV/
│
├── fedprism/
│   ├── server.py          # FedPrism server logic
│   ├── client.py          # Local training + delta computation
│   ├── clustering.py      # Covariance / Hierarchical clustering
│   ├── personalization.py # Alpha updates + soft assignment
│   ├── utils.py
│
├── yolo_experiments/
│   ├── train_fed_yolo.py  # YOLOv5/v8/v12 FedAvg training
│   ├── configs/
│   └── results/
│
├── datasets/
│
├── results/
│   ├── YOLO_comparison/
│   ├── FedPrism_vs_FedClust/
│   └── multi_dataset_validation/
│
└── README.md
```

---

## ⚙️ Installation

### Requirements

* Python 3.10+
* PyTorch 2.0+
* CUDA 11.8
* Ultralytics YOLOv5/v8/v12
* scikit-learn
* numpy, matplotlib

### Setup

```bash
git clone https://github.com/<username>/FedPrism-IoV
cd FedPrism-IoV
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### **1. Run YOLO Federated Training**

```bash
python yolo_experiments/train_fed_yolo.py --model yolov12 --rounds 10
```

### **2. Run FedPrism**

```bash
python fedprism/server.py --config configs/fedprism.yaml
```

### **3. Run FedClust Baseline**

```bash
python baselines/fedclust.py --config configs/fedclust.yaml
```

---

## 📊 Experimental Settings

### YOLO Benchmark

* Dataset: **COCO128**
* Clients: **8**
* FL Strategy: **FedAvg**
* 10 rounds × 2 local epochs

### FedPrism Benchmark

* Datasets: **MNIST, CIFAR-10, Fashion-MNIST, CIFAR-100**
* 100 clients (10% participation)
* Non-IID: Dirichlet α ∈ {0.1, 0.3, 0.5}
* Clustering every 10 rounds
* Top-M soft assignment

---

## 🧠 Why FedPrism Works

✔ Learns how much global knowledge a client needs
✔ Soft clustering captures hybrid client distributions
✔ Dynamic re-clustering adapts to concept drift
✔ More clusters → finer specialization
✔ Personalized model per client → higher accuracy

---

## ⚠ Limitations

* YOLOv12 is heavier → higher latency
* FedPrism sensitive to hyperparameters
* Adds computational overhead (dynamic clustering)
* Needs real-world IoV dataset validation

---

## 📌 Future Work

* Combine FedPrism + YOLOv12 for full IoV deployment
* Auto-tune α via meta-learning
* Add differential privacy
* Gradient compression for edge devices
* Validate on **nuScenes / Waymo / BDD100K**

---

## 👥 Authors

* **Prakash Kumbhakar** — FedPrism algorithm, implementation, analysis
* **Shrey Srivastava** — YOLO benchmarking, FedClust baseline, visualization

---
