import copy
import torch
from .base import BaseServer, LocalUpdate

class LocalServer(BaseServer):
    def __init__(self, args, dataset, dict_users, model, test_dataset=None, dict_users_test=None):
        super().__init__(args, dataset, dict_users, test_dataset, dict_users_test)
        self.global_model = copy.deepcopy(model)
        self.global_model.to(self.args.device)

    def train(self):
        loss_train = []
        acc_test = []
        acc_test_local = []
        
        # For Local, we train a separate model for each client
        # We can just train them sequentially or in parallel
        # Since we don't aggregate, we just track average performance
        
        # Initialize client models
        client_models = {i: copy.deepcopy(self.global_model) for i in range(self.args.num_users)}
        
        for iter in range(self.args.epochs):
            loss_locals = []
            
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = torch.randperm(self.args.num_users)[:m]
            
            self.log(f"Round {iter}: Selected clients: {idxs_users.tolist()}")
            
            for idx in idxs_users:
                client_idx = idx.item()
                self.log(f"  Training client {client_idx}...")
                
                net = client_models[client_idx]
                net.to(self.args.device)
                
                local = LocalUpdate(args=self.args, dataset=self.dataset, idxs=self.dict_users[client_idx])
                w, loss = local.train(net=net)
                
                # Update client model
                net.load_state_dict(w)
                loss_locals.append(loss)
                self.log(f"  Client {client_idx} loss: {loss:.4f}")
                
            loss_avg = sum(loss_locals) / len(loss_locals)
            
            # Test accuracy (Global - Average of local models?)
            # For "Local", there is no global model. 
            # But we can evaluate the ensemble or just average local accuracy.
            # Let's report Average Local Accuracy as "Global" for consistency in plots?
            # Or evaluate each local model on the GLOBAL test set and average that?
            
            # 1. Average Local Accuracy (on local test sets)
            acc_local = self.test_on_clients(client_models)
            acc_test_local.append(acc_local)
            
            # 2. Average Global Accuracy (each local model on global test set)
            acc_globals = []
            for i in range(self.args.num_users):
                net = client_models[i]
                acc, _ = self.test(net, self.test_dataset, self.args)
                acc_globals.append(acc)
            acc_global_avg = sum(acc_globals) / len(acc_globals)
            acc_test.append(acc_global_avg)
            
            self.log(f"Round {iter}, Loss: {loss_avg:.4f}, Global Acc (Avg): {acc_global_avg:.2f}%, Local Acc: {acc_local:.2f}%")
            loss_train.append(loss_avg)
            
        return client_models, loss_train, acc_test, acc_test_local
