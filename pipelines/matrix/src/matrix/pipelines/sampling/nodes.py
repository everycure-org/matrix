import logging
from typing import Tuple

import pyspark.sql as ps

from matrix.inject import inject_object
from matrix.pipelines.sampling.samplers import Sampler

logger = logging.getLogger(__name__)


@inject_object()
def sample_nodes(
    sampler: Sampler,
    kg_nodes: ps.DataFrame,
    kg_edges: ps.DataFrame,
    gt_p: ps.DataFrame,
    gt_n: ps.DataFrame,
    embeddings_nodes: ps.DataFrame,
) -> Tuple[ps.DataFrame, ps.DataFrame, ps.DataFrame, ps.DataFrame]:
    return sampler.sample(kg_nodes, kg_edges, gt_p, gt_n, embeddings_nodes)
