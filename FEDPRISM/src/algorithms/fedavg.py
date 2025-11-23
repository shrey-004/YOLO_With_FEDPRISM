import copy
import torch
from .base import BaseServer, LocalUpdate

class FedAvgServer(BaseServer):
    def __init__(self, args, dataset, dict_users, model, test_dataset=None, dict_users_test=None):
        super().__init__(args, dataset, dict_users, test_dataset, dict_users_test)
        self.global_model = copy.deepcopy(model)
        self.global_model.to(self.args.device)

    def train(self):
        loss_train = []
        acc_test = []
        acc_test_local = []
        
        for iter in range(self.args.epochs):
            self.global_model.train() # Ensure model is in training mode for local updates
            w_locals, loss_locals = [], []
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = torch.randperm(self.args.num_users)[:m] # Randomly sample clients
            
            self.log(f"Round {iter}: Selected clients: {idxs_users.tolist()}")
            
            for idx in idxs_users:
                client_idx = idx.item()
                self.log(f"  Training client {client_idx}...")
                local = LocalUpdate(args=self.args, dataset=self.dataset, idxs=self.dict_users[client_idx])
                w, loss = local.train(net=copy.deepcopy(self.global_model).to(self.args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                self.log(f"  Client {client_idx} loss: {loss:.4f}")
                
            # Update global weights
            w_glob = self.aggregate(w_locals)
            self.global_model.load_state_dict(w_glob)
            
            loss_avg = sum(loss_locals) / len(loss_locals)
            
            # Test accuracy (Global)
            self.global_model.eval()
            acc, _ = self.test(self.global_model, self.test_dataset, self.args) # Use full test dataset
            acc_test.append(acc)
            
            # Test accuracy (Local)
            acc_local = self.test_on_clients() # Uses global model on local test sets
            acc_test_local.append(acc_local)
            
            self.log(f"Round {iter}, Loss: {loss_avg:.4f}, Global Acc: {acc:.2f}%, Local Acc: {acc_local:.2f}%")
            loss_train.append(loss_avg)
            
        return self.global_model, loss_train, acc_test, acc_test_local
