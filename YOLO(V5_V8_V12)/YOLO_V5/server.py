# import flwr as fl
# from typing import List, Tuple, Dict

# # ✅ Weighted average for federated metric aggregation (aggregates 'loss')
# def weighted_average_loss(metrics: List[Tuple[int, Dict[str, float]]]):
#     """
#     Aggregates the 'loss' metric from clients.
#     metrics: list of tuples like [(num_examples, metrics_dict), ...]
#     """
#     total_loss = 0.0
#     total_examples = 0
    
#     # metrics is a list of (num_examples, metrics_dict)
#     for num_examples, m in metrics:
#         if "loss" in m:
#             total_loss += m["loss"] * num_examples
#         total_examples += num_examples

#     if total_examples == 0:
#         return {"avg_loss": 0.0}

#     avg_loss = total_loss / total_examples
#     print(f"Server-side evaluation: avg_loss = {avg_loss}")
#     return {"avg_loss": avg_loss}  # ✅ Return dict with aggregated loss


# # ✅ Start Flower server
# fl.server.start_server(
#     server_address="localhost:8080",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=fl.server.strategy.FedAvg(
#         # Use the new aggregation function for 'evaluate' metrics
#         evaluate_metrics_aggregation_fn=weighted_average_loss
#     ),
# )



import flwr as fl
from typing import List, Tuple, Dict

# ✅ Weighted average for federated metric aggregation (aggregates 'loss')
def weighted_average_loss(metrics: List[Tuple[int, Dict[str, float]]]):
    """
    Aggregates the 'loss' metric from clients.
    metrics: list of tuples like [(num_examples, metrics_dict), ...]
    """
    total_loss = 0.0
    total_examples = 0
    
    # metrics is a list of (num_examples, metrics_dict)
    for num_examples, m in metrics:
        if "loss" in m:
            total_loss += m["loss"] * num_examples
        total_examples += num_examples

    if total_examples == 0:
        return {"avg_loss": 0.0}

    avg_loss = total_loss / total_examples
    print(f"Server-side evaluation: avg_loss = {avg_loss}")
    return {"avg_loss": avg_loss}  # ✅ Return dict with aggregated loss


# --- Strategy Configuration ---
# Configure the FedAvg strategy to use 8 clients
strategy = fl.server.strategy.FedAvg(
    min_available_clients=8,     # Wait for all 8 clients to connect
    min_fit_clients=8,           # Use all 8 clients for training in each round
    min_evaluate_clients=8,      # Use all 8 clients for evaluation in each round
    evaluate_metrics_aggregation_fn=weighted_average_loss  # Your custom aggregation
)

print("Starting server, waiting for 8 clients...")

# ✅ Start Flower server with the new 8-client strategy
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy  # Pass the configured strategy
)