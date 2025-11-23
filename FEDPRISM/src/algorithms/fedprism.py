import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseServer, LocalUpdate

class PrismLocalUpdate(LocalUpdate):
    def __init__(self, args, dataset, idxs, alpha_init=0.5):
        super().__init__(args, dataset, idxs)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, device=args.device))
        
    def train_prism(self, net_glob, net_clust):
        # Placeholder for potential future gradient-based alpha updates
        pass

class FedPrismServer(BaseServer):
    def __init__(self, args, dataset, dict_users, model, test_dataset=None, dict_users_test=None):
        super().__init__(args, dataset, dict_users, test_dataset, dict_users_test)
        self.global_model = copy.deepcopy(model)
        self.global_model.to(self.args.device)
        
        self.K = args.num_clusters
        self.m = args.num_assignments
        self.alpha = args.alpha_coef
        self.clustering_method = args.clustering_method 
        self.trainable_alpha = args.trainable_alpha
        
        self.cluster_models = {c: copy.deepcopy(self.global_model.state_dict()) for c in range(self.K)}
        self.W = {i: {c: 1.0/self.K for c in range(self.K)} for i in range(self.args.num_users)}
        
        # Per-client alpha
        self.client_alphas = {i: self.alpha for i in range(self.args.num_users)}
        
        self.client_features = {} 

    def _flatten_weights(self, w):
        return torch.cat([p.flatten() for p in w.values()]).cpu().numpy()

    def _get_personalized_model(self, client_idx):
        w_g = self.global_model.state_dict()
        w_pers = copy.deepcopy(w_g)
        
        cluster_sum = None
        weights = self.W[client_idx]
        
        for c, weight in weights.items():
            if weight > 0:
                w_c = self.cluster_models[c]
                if cluster_sum is None:
                    cluster_sum = {k: weight * w_c[k] for k in w_c}
                else:
                    for k in w_c:
                        cluster_sum[k] += weight * w_c[k]
                        
        if cluster_sum is None:
             return w_g

        # Use client-specific alpha
        alpha = self.client_alphas[client_idx]
        
        for k in w_pers:
            w_pers[k] = alpha * w_g[k] + (1 - alpha) * cluster_sum[k]
            
        return w_pers

    def _update_clusters(self, active_clients):
        valid_clients = [c for c in active_clients if c in self.client_features]
        if len(valid_clients) < self.K:
            return 
            
        features = np.array([self.client_features[c] for c in valid_clients])
        
        labels = None
        centroids = None
        
        if self.clustering_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.K, n_init=10).fit(features)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            
        elif self.clustering_method in ['single', 'average']:
            Z = linkage(features, method=self.clustering_method, metric='euclidean')
            labels = fcluster(Z, t=self.K, criterion='maxclust') - 1 
            
            centroids = np.zeros((self.K, features.shape[1]))
            for c in range(self.K):
                cluster_feats = features[labels == c]
                if len(cluster_feats) > 0:
                    centroids[c] = np.mean(cluster_feats, axis=0)
                    
        elif self.clustering_method == 'covariance':
            sim_matrix = cosine_similarity(features)
            from sklearn.cluster import SpectralClustering
            sc = SpectralClustering(n_clusters=self.K, affinity='precomputed', n_init=10)
            labels = sc.fit_predict((sim_matrix + 1) / 2) 
            
            centroids = np.zeros((self.K, features.shape[1]))
            for c in range(self.K):
                cluster_feats = features[labels == c]
                if len(cluster_feats) > 0:
                    centroids[c] = np.mean(cluster_feats, axis=0)

        for i, client_idx in enumerate(valid_clients):
            client_feat = features[i]
            sims = []
            for c in range(self.K):
                sim = cosine_similarity(client_feat.reshape(1, -1), centroids[c].reshape(1, -1))[0][0]
                sims.append((c, sim))
            
            sims.sort(key=lambda x: x[1], reverse=True)
            top_m = sims[:self.m]
            
            exps = [np.exp(s[1]) for s in top_m]
            sum_exps = sum(exps)
            
            new_weights = {c: 0.0 for c in range(self.K)}
            for idx, (c, _) in enumerate(top_m):
                new_weights[c] = exps[idx] / sum_exps
                
            self.W[client_idx] = new_weights

    def train(self):
        loss_train = []
        acc_test = []
        acc_test_local = []
        alpha_history = [] # Track alpha mean
        
        for iter in range(self.args.epochs):
            w_locals = {}
            loss_locals = []
            
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = torch.randperm(self.args.num_users)[:m]
            
            self.log(f"Round {iter}: Selected clients: {idxs_users.tolist()}")
            
            # 1. Distribute & Train
            for idx in idxs_users:
                client_idx = idx.item()
                self.log(f"  Client {client_idx} training...")
                
                # Generate personalized model
                w_pers = self._get_personalized_model(client_idx)
                net = copy.deepcopy(self.global_model)
                net.load_state_dict(w_pers)
                net.to(self.args.device)
                
                local = LocalUpdate(args=self.args, dataset=self.dataset, idxs=self.dict_users[client_idx])
                w, loss = local.train(net=net)
                
                # Trainable Alpha Update Logic
                if self.trainable_alpha:
                    # 1. Eval Global
                    net_g = copy.deepcopy(self.global_model)
                    net_g.load_state_dict(self.global_model.state_dict())
                    net_g.to(self.args.device)
                    _, loss_g = self.test(net_g, local.ldr_train.dataset, self.args) # Use train set for speed
                    
                    # 2. Eval Cluster (Weighted)
                    # Construct cluster model
                    w_c_combined = None
                    weights = self.W[client_idx]
                    total_w = 0
                    for c, weight in weights.items():
                        if weight > 0:
                            w_c = self.cluster_models[c]
                            if w_c_combined is None:
                                w_c_combined = {k: weight * w_c[k] for k in w_c}
                            else:
                                for k in w_c:
                                    w_c_combined[k] += weight * w_c[k]
                            total_w += weight
                    
                    if w_c_combined:
                        for k in w_c_combined:
                            w_c_combined[k] = torch.div(w_c_combined[k], total_w)
                        
                        net_c = copy.deepcopy(self.global_model)
                        net_c.load_state_dict(w_c_combined)
                        net_c.to(self.args.device)
                        _, loss_c = self.test(net_c, local.ldr_train.dataset, self.args)
                        
                        # Update Alpha
                        current_alpha = self.client_alphas[client_idx]
                        if loss_g < loss_c:
                            new_alpha = min(1.0, current_alpha + 0.05)
                        else:
                            new_alpha = max(0.0, current_alpha - 0.05)
                            
                        self.client_alphas[client_idx] = new_alpha
                        self.log(f"    Client {client_idx} Alpha updated: {current_alpha:.2f} -> {new_alpha:.2f} (L_g={loss_g:.4f}, L_c={loss_c:.4f})")

                w_locals[client_idx] = w
                loss_locals.append(loss)
                self.log(f"  Client {client_idx} loss: {loss:.4f}")
                
                if self.clustering_method == 'covariance':
                    flat_w = self._flatten_weights(w)
                    flat_start = self._flatten_weights(w_pers)
                    self.client_features[client_idx] = flat_w - flat_start
                else:
                    self.client_features[client_idx] = self._flatten_weights(w)

            # Aggregation
            w_glob_new = self.aggregate(list(w_locals.values()))
            self.global_model.load_state_dict(w_glob_new)
            
            for c in range(self.K):
                w_c_new = None
                total_weight = 0
                
                for client_idx, w_client in w_locals.items():
                    weight = self.W[client_idx].get(c, 0)
                    if weight > 0:
                        if w_c_new is None:
                            w_c_new = {k: weight * w_client[k] for k in w_client}
                        else:
                            for k in w_client:
                                w_c_new[k] += weight * w_client[k]
                        total_weight += weight
                
                if w_c_new is not None and total_weight > 0:
                    for k in w_c_new:
                        w_c_new[k] = torch.div(w_c_new[k], total_weight)
                    self.cluster_models[c] = w_c_new
            
            if (iter + 1) % self.args.clustering_freq == 0:
                self._update_clusters(list(w_locals.keys()))
            
            loss_avg = sum(loss_locals) / len(loss_locals)
            
            # Test accuracy (Global)
            self.global_model.eval()
            acc, _ = self.test(self.global_model, self.test_dataset, self.args)
            acc_test.append(acc)
            
            # Test accuracy (Local Personalized)
            models_dict = {}
            for i in range(self.args.num_users):
                w_pers = self._get_personalized_model(i)
                net = copy.deepcopy(self.global_model)
                net.load_state_dict(w_pers)
                models_dict[i] = net
                
            acc_local = self.test_on_clients(models_dict)
            acc_test_local.append(acc_local)
            
            self.log(f"Round {iter}, Loss: {loss_avg:.4f}, Global Acc: {acc:.2f}%, Local Acc: {acc_local:.2f}%")
            loss_train.append(loss_avg)
            
            # Track average alpha
            avg_alpha = sum(self.client_alphas.values()) / len(self.client_alphas)
            alpha_history.append(avg_alpha)
            
        return self.global_model, loss_train, alpha_history, acc_test, acc_test_local
