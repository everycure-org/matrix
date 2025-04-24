from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matrix.pipelines.matrix_generation.reporting_plots import (
    MultiScoreHistogram,
    SingleScoreHistogram,
    SingleScoreLinePlot,
    SingleScoreScatterPlot,
)


@pytest.fixture
def sample_matrix_data():
    """Fixture that provides sample matrix data with multiple scores for testing."""
    return pd.DataFrame(
        {
            "source": ["drug_1", "drug_2", "drug_3", "drug_4"],
            "target": ["disease_1", "disease_2", "disease_3", "disease_4"],
            "score_1": [0.9, 0.7, 0.5, 0.3],
            "score_2": [0.85, 0.65, 0.45, 0.25],
        }
    )


def test_plot_generator(sample_matrix_data):
    """This test all plotting strategies."""
    # Given any plotting strategy
    generators = [
        SingleScoreHistogram(
            name="name",
            score_col="score_1",
            is_log_y_scale=True,
            figsize=(10, 6),
        ),
        MultiScoreHistogram(
            name="name",
            score_cols_lst=["score_1", "score_2"],
        ),
        SingleScoreLinePlot(
            name="name",
            score_col="score_1",
            is_log_y_scale=True,
            figsize=(10, 6),
        ),
        SingleScoreScatterPlot(
            name="name",
            score_col="score_1",
            n_sample=1000000,
            figsize=(10, 6),
            points_alpha=0.03,
            points_s=0.5,
        ),
    ]

    for generator in generators:
        # When generating the plot
        plot = generator.generate(sample_matrix_data)
        # Then:
        # The plot is a matplotlib figure
        assert isinstance(plot, plt.Figure)
        # The generator has the correct name
        assert generator.name == "name"
