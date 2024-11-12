from kedro.pipeline import Pipeline, node, pipeline
import pandas as pd
import torch


def check_gpu_availability():
    if torch.cuda.is_available():
        return pd.DataFrame({"gpu": ["available"]})
    else:
        raise ValueError("No GPU available")


def create_pipeline(**kwargs) -> Pipeline:
    """Create GPU validation pipeline."""
    return pipeline(
        [
            node(
                func=check_gpu_availability,
                inputs={},
                outputs={"gpu_validation": "gpu_validation"},
                name="check_gpu_availability",
            ),
        ]
    )
