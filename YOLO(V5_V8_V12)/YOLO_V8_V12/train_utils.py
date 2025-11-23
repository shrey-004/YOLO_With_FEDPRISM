import torch
from tqdm import tqdm

def compute_loss(preds, labels):
    """
    Dummy loss that handles a list of 3 outputs from the model.
    """
    # Check if preds is a list (from YOLOv12Scratch)
    if isinstance(preds, list):
        total_loss = 0
        for pred in preds:
            # A simple dummy loss for each of the 3 output heads
            total_loss += ((pred.mean() - 0.5) ** 2).mean()
        return total_loss / len(preds) # Average the loss of the 3 heads
    
    # Fallback for a model with a single output
    else:
        return ((preds.mean() - 0.5) ** 2).mean()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    # The collate_fn gives (imgs, labels_list)
    for imgs, labels in tqdm(dataloader, desc="Training"):
        imgs = imgs.to(device)
        # Labels are a list, so they stay on the CPU
        
        optimizer.zero_grad()
        
        # Model forward pass
        pred = model(imgs)
        
        # Loss computation
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
            imgs = imgs.to(device)
            
            pred = model(imgs)
            
            loss = compute_loss(pred, labels)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)