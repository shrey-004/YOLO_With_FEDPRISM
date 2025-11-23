import flwr as fl
import torch
from torch.utils.data import DataLoader
# FIX 1: Make sure your model file is named 'YOLOv8Scratch.py'
from YOLOV8scratch import YOLOv8Scratch
from dataset_utils import YOLODataset, yolo_collate_fn
# These utils MUST be the new versions that handle the 3-output list
from train_utils import train_one_epoch, evaluate
import os

CLIENT_ID = int(os.environ.get("CLIENT_ID", 1))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = f"Federated_COCO128/client_{CLIENT_ID}/"
EPOCHS = 2
BATCH_SIZE = 2

# FIX 2: You must instantiate the model you imported
model = YOLOv8Scratch(num_classes=80).to(DEVICE)

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