"""Utils for inference running."""

from typing import Dict
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_kdeplot(scores: pd.DataFrame, infer_type: Dict, col_name: str) -> plt.figure:
    """Create visualisations based on the treat scores.

    Args:
        scores: treat scores generated during the inference.
        infer_type: type of inference requested.
        col_name: column name to be used for plots.

    Returns:
        figure: figure saved locally and in MLFlow
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.kdeplot(scores[col_name])
    ax.set_title(
        f"Distribution of Treatment Scores; \n {infer_type['request']}",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Treatment Score", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)

    # Add gridlines for better readability
    ax.grid(True, linestyle="--", alpha=0.7)
    caption = (
        f"Mean: {np.mean(scores[col_name]):.2f}, Std: {np.std(scores[col_name]):.2f}, "
        f"Min: {min(scores[col_name]):.2f}, Max: {max(scores[col_name]):.2f}"
    )

    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=14, fontstyle="italic")
    plt.figtext(0.9, 0.01, infer_type["time"], ha="left", fontsize=14, fontstyle="italic")

    return fig
