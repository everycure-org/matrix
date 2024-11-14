from kedro.pipeline import Pipeline, pipeline
import pandas as pd
import torch

from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig


def check_gpu_availability():
    if torch.cuda.is_available():
        return pd.DataFrame({"gpu": ["available"]})
    else:
        raise ValueError("No GPU available")


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
