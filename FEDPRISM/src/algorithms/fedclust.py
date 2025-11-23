import copy
import torch
import numpy as np
from sklearn.cluster import KMeans
from .base import BaseServer, LocalUpdate

class FedClustServer(BaseServer):
    def __init__(self, args, dataset, dict_users, model, test_dataset=None, dict_users_test=None):
        super().__init__(args, dataset, dict_users, test_dataset, dict_users_test)
        self.global_model = copy.deepcopy(model)
        self.global_model.to(self.args.device)
        self.cluster_models = {} # Map cluster_id -> model state_dict
        self.client_clusters = {i: 0 for i in range(self.args.num_users)} # Map client_id -> cluster_id
        self.n_clusters = args.num_clusters # C in the paper
        
        # Initialize cluster models (initially all same as global)
        for c in range(self.n_clusters):
            self.cluster_models[c] = copy.deepcopy(self.global_model.state_dict())

    def _flatten_weights(self, w):
        return torch.cat([p.flatten() for p in w.values()]).cpu().numpy()

    def train(self):
        loss_train = []
        acc_test = []
        acc_test_local = []
        
        for iter in range(self.args.epochs):
            w_locals = {} # Map client_id -> w
            loss_locals = []
            
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = torch.randperm(self.args.num_users)[:m]
            
            self.log(f"Round {iter}: Selected clients: {idxs_users.tolist()}")
            
            # 1. Distribute models
            for idx in idxs_users:
                client_idx = idx.item()
                cluster_id = self.client_clusters[client_idx]
                
                self.log(f"  Client {client_idx} (Cluster {cluster_id}) training...")
                
                # Load the cluster specific model
                net = copy.deepcopy(self.global_model)
                net.load_state_dict(self.cluster_models[cluster_id])
                net.to(self.args.device)
                
                local = LocalUpdate(args=self.args, dataset=self.dataset, idxs=self.dict_users[client_idx])
                w, loss = local.train(net=net)
                
                w_locals[client_idx] = w
                loss_locals.append(loss)
                self.log(f"  Client {client_idx} loss: {loss:.4f}")
            
            # 2. Clustering (if needed, e.g., every round or periodically)
            # FedClust paper says: Server clusters clients based on similarity of w_k
            # We cluster ALL active clients this round
            
            active_clients = list(w_locals.keys())
            if len(active_clients) >= self.n_clusters:
                flat_weights = np.array([self._flatten_weights(w_locals[c]) for c in active_clients])
                
                # Clustering
                # Using K-Means as default or Hierarchical
                clustering = KMeans(n_clusters=self.n_clusters, n_init=10).fit(flat_weights)
                labels = clustering.labels_
                
                # Update client_clusters map for these active clients
                # Note: In FedClust, clusters might be dynamic per round or persistent.
                # The paper implies dynamic grouping.
                # We need to map the NEW labels to the stored cluster models.
                # This is tricky because label '0' in round t might not be label '0' in round t+1.
                # However, the algorithm description says: "Server Aggregates w_k within each cluster C_c to form w_c_new"
                # So we just form NEW cluster models for THIS round's clusters.
                
                new_cluster_models = {}
                
                # Aggregate within new clusters
                for c in range(self.n_clusters):
                    cluster_clients = [active_clients[i] for i in range(len(active_clients)) if labels[i] == c]
                    if cluster_clients:
                        cluster_weights = [w_locals[cli] for cli in cluster_clients]
                        new_cluster_models[c] = self.aggregate(cluster_weights)
                        
                        # Update client mapping
                        for cli in cluster_clients:
                            self.client_clusters[cli] = c
                    else:
                        # If a cluster is empty, maybe keep old one or re-init?
                        # For simplicity, keep old one if it exists, or global
                        if c in self.cluster_models:
                            new_cluster_models[c] = self.cluster_models[c]
                        else:
                            new_cluster_models[c] = self.global_model.state_dict()

                self.cluster_models = new_cluster_models
                
            else:
                # Not enough clients to cluster, just aggregate all to global and assign to all clusters
                w_avg = self.aggregate(list(w_locals.values()))
                for c in range(self.n_clusters):
                    self.cluster_models[c] = w_avg
            
            loss_avg = sum(loss_locals) / len(loss_locals)
            
            # Test accuracy (Global Model - which one? Maybe average of clusters?)
            w_all_clusters = list(self.cluster_models.values())
            w_glob = self.aggregate(w_all_clusters)
            self.global_model.load_state_dict(w_glob)
            
            self.global_model.eval()
            acc, _ = self.test(self.global_model, self.test_dataset, self.args)
            acc_test.append(acc)
            
            # Test accuracy (Local)
            # For FedClust, clients should use their CLUSTER model
            models_dict = {}
            for i in range(self.args.num_users):
                cluster_id = self.client_clusters[i]
                net = copy.deepcopy(self.global_model)
                net.load_state_dict(self.cluster_models[cluster_id])
                models_dict[i] = net
            
            acc_local = self.test_on_clients(models_dict)
            acc_test_local.append(acc_local)
            
            self.log(f"Round {iter}, Loss: {loss_avg:.4f}, Global Acc: {acc:.2f}%, Local Acc: {acc_local:.2f}%")
            loss_train.append(loss_avg)
            
        return self.cluster_models, loss_train, acc_test, acc_test_local
