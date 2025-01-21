import logging
from typing import Tuple

import pyspark.sql as ps

from matrix.inject import inject_object
from matrix.pipelines.create_sample.samplers import Sampler

logger = logging.getLogger(__name__)


@inject_object()
def sample_nodes(
    sampler: Sampler,
    knowledge_graph_nodes: ps.DataFrame,
    knowledge_graph_edges: ps.DataFrame,
    ground_truth_edges: ps.DataFrame,
) -> Tuple[ps.DataFrame, ps.DataFrame]:
    return sampler.sample(knowledge_graph_nodes, knowledge_graph_edges, ground_truth_edges)
