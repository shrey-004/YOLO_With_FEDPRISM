import flwr as fl
import torch
from torch.utils.data import DataLoader
from YOLOModernScratch import YOLOModernScratch   # <-- FIX 1: Changed file and class name
from dataset_utils import YOLODataset, yolo_collate_fn  # ✅ added here
from train_utils import train_one_epoch, evaluate
import os
import torch.nn as nn
model = YOLOModernScratch(num_classes=80)
model.apply(lambda m: isinstance(m, nn.Conv2d) and nn.init.xavier_uniform_(m.weight))
# ✅ ADD THIS LINE BACK
CLIENT_ID = int(os.environ.get("CLIENT_ID", 1))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = f"Federated_COCO128/client_{CLIENT_ID}/"
EPOCHS = 2
BATCH_SIZE = 2

model = YOLOModernScratch(num_classes=80).to(DEVICE)  # <-- FIX 2

# train_dataset = YOLODataset(
# ... (rest of file is fine)

train_dataset = YOLODataset(
    os.path.join(DATA_PATH, "data.yaml"),
    os.path.join(DATA_PATH, "images/train"),
    os.path.join(DATA_PATH, "labels/train")
)
val_dataset = YOLODataset(
    os.path.join(DATA_PATH, "data.yaml"),
    os.path.join(DATA_PATH, "images/val"),
    os.path.join(DATA_PATH, "labels/val")
)

# ✅ Use the custom collate function
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yolo_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yolo_collate_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class YOLOClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_one_epoch(model, train_loader, optimizer, DEVICE)
        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = evaluate(model, val_loader, DEVICE)
        return float(loss), len(val_loader.dataset), {"loss": float(loss)}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=YOLOClient())


