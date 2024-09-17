"""Utils for inference running."""
from abc import ABC, abstractmethod
from typing import List, Type
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ..modelling.model import ModelWrapper


def create_kdeplot(scores: pd.DataFrame, infer_type: str):
    """Create visualisations based on the treat scores and store them in GCS/MLFlow.

    Args:
        scores: treat scores generated during the inference.
        infer_type: type of inference requested.

    Returns:
        figure: figure saved locally and in MLFlow
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.kdeplot(scores["treat score"])
    ax.set_title(
        f"Distribution of Treatment Scores; {infer_type}",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xlabel("Treatment Score", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)

    # Add gridlines for better readability
    ax.grid(True, linestyle="--", alpha=0.7)
    caption = (
        f"Mean: {np.mean(scores['treat score']):.2f}, Std: {np.std(scores['treat score']):.2f}, "
        f"Min: {min(scores['treat score']):.2f}, Max: {max(scores['treat score']):.2f}"
    )

    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=14, fontstyle="italic")

    return fig
