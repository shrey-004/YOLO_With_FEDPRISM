import torch
from tqdm import tqdm

def compute_loss(pred, labels):
    # Dummy loss (for demo) — replace with real YOLOv5 loss
    return ((pred.mean() - 0.5) ** 2).mean()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(dataloader, desc="Training"):
        # imgs, labels = imgs.to(device), labels.to(device)
        imgs = imgs.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = compute_loss(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating"):
            # imgs, labels = imgs.to(device), labels.to(device)
            imgs = imgs.to(device)
            pred = model(imgs)
            total_loss += compute_loss(pred, labels).item()
    return total_loss / len(dataloader)