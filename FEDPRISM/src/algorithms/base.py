import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class BaseServer(object):
    def __init__(self, args, dataset, dict_users, test_dataset=None, dict_users_test=None):
        self.args = args
        self.dataset = dataset
        self.dict_users = dict_users
        self.test_dataset = test_dataset
        self.dict_users_test = dict_users_test
        self.global_model = None # To be initialized in child classes
        self.verbose = True # Enable verbose logging

    def log(self, msg):
        if self.verbose:
            print(msg)

    def aggregate(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg

    def test(self, net, dataset, args):
        net.eval()
        # use the provided dataset for testing
        data_loader = DataLoader(dataset, batch_size=args.bs)
        correct = 0
        total = 0
        loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        return acc, loss

    def test_on_clients(self, models_dict=None):
        """
        Evaluate models on their respective client's local test set.
        If models_dict is None, use global_model for all clients.
        """
        if self.dict_users_test is None or self.test_dataset is None:
            return 0.0
            
        acc_locals = []
        
        # We evaluate on ALL clients or a subset? 
        # Evaluating on ALL clients is best for benchmark.
        
        for idx in range(self.args.num_users):
            if models_dict:
                net = models_dict[idx]
            else:
                net = self.global_model
                
            # Create local test loader
            idxs_test = self.dict_users_test[idx]
            if len(idxs_test) == 0:
                continue
                
            local_test_data = DatasetSplit(self.test_dataset, idxs_test)
            acc, _ = self.test(net, local_test_data, self.args)
            acc_locals.append(acc)
            
        return sum(acc_locals) / len(acc_locals) if acc_locals else 0.0
