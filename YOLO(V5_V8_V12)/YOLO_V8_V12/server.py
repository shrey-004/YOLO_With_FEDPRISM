import flwr as fl
from typing import List, Tuple, Dict
from YOLOModernScratch import YOLOModernScratch   # <-- FIX 1: Changed file and class name
# from YOLOV8scratch import YOLOv8Scratch

import torch.nn as nn
model = YOLOModernScratch(num_classes=80)
model.apply(lambda m: isinstance(m, nn.Conv2d) and nn.init.xavier_uniform_(m.weight))

# ✅ Aggregates the 'loss' metric from clients
def weighted_average_loss(metrics: List[Tuple[int, Dict[str, float]]]):
    """
    Aggregates the 'loss' metric from clients.
    metrics: list of tuples like [(num_examples, metrics_dict), ...]
    """
    total_loss = 0.0
    total_examples = 0
    
    for num_examples, m in metrics:
        if "loss" in m:
            total_loss += m["loss"] * num_examples
        total_examples += num_examples

    if total_examples == 0:
        return {"avg_loss": 0.0} # Return a dict

    avg_loss = total_loss / total_examples
    print(f"Server-side evaluation: avg_loss = {avg_loss}")
    return {"avg_loss": avg_loss}  # ✅ Return dict with aggregated loss


# --- Strategy Configuration ---
model = YOLOModernScratch(num_classes=80)
initial_parameters = fl.common.ndarrays_to_parameters(
    [v.cpu().numpy() for v in model.state_dict().values()]
)

strategy = fl.server.strategy.FedAvg(
    initial_parameters=initial_parameters,
    min_available_clients=8,
    min_fit_clients=8,
    min_evaluate_clients=8,
)

print("Starting server, waiting for 8 clients...")

# ✅ Start Flower server with the new 8-client strategy
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy  # Pass the configured strategy
)