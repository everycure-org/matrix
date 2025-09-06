import logging
from typing import List

from evaluation import plot_av_ranking_metrics

logger = logging.getLogger(__name__)


def recall_at_n_plots(
    matrices: List[ps.DataFrame],
    model_names: List[str],
) -> ps.DataFrame:
    """Function to calculate recall at n."""

    return plot_av_ranking_metrics(
        matrices_all=matrices,
        model_names=model_names,
        bool_test_col="is_known_positive",
        score_col="treat score",
        perform_sort=False,
        sup_title="Standard positive ground truth",
        is_average_folds=True,
    )
