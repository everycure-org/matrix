import logging
import time
from kedro.pipeline import Pipeline, pipeline
import pandas as pd
import torch

from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig


def check_gpu_availability() -> pd.DataFrame:
    if torch.cuda.is_available():
        return pd.DataFrame({"gpu": ["available"]})
    else:
        raise ValueError("No GPU available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        status = "available"
    else:
        print("No GPU available, falling back to CPU")
        status = "unavailable (using CPU)"

    # Create a simple synthetic dataset
    X = torch.randn(500000, 1000, device=device)  # Increased dimensions
    y = (X.sum(dim=1) > 0).float()

    time_start = time.time()
    # Define a larger neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
        torch.nn.Sigmoid(),
    ).to(device)

    # Training setup
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop with time check
    max_time = 600  # 60 seconds = 1 minute
    epoch = 0
    while time.time() - time_start < max_time:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        epoch += 1

        if epoch % 5 == 0:  # Log progress every 5 epochs
            logging.info(f"Completed {epoch} epochs. Time elapsed: {time.time() - time_start:.2f}s")

    # Make predictions
    with torch.no_grad():
        predictions = model(X[:5])

    # Return results
    result_dict = {
        "gpu": [status],
        "final_loss": [float(loss)],
        "sample_predictions": [predictions.cpu().numpy().tolist()],  # predictions already moved to CPU
    }

    time_end = time.time()
    result_dict["training_time"] = [time_end - time_start]

    logging.info(f"Training time: {time_end - time_start} seconds")

    if device.type == "cuda":
        result_dict["cuda_device"] = [torch.cuda.get_device_name()]
    else:
        result_dict["cuda_device"] = ["CPU"]

    return result_dict


def create_pipeline(**kwargs) -> Pipeline:
    """Create GPU validation pipeline."""
    return pipeline(
        [
            ArgoNode(
                func=check_gpu_availability,
                inputs={},
                outputs={"gpu_validation": "gpu_validation"},
                name="check_gpu_availability",
                argo_config=ArgoResourceConfig(
                    num_gpus=1,
                    memory_limit=32,
                    memory_request=16,
                    cpu_limit=4,
                    cpu_request=4,
                ),
            ),
        ]
    )
