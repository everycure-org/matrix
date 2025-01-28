import logging
from typing import Dict

import pyspark.sql as ps

from matrix.inject import inject_object
from matrix.pipelines.create_sample.samplers import Sampler
from matrix.utils.pa_utils import Column, DataFrameSchema, check_output

logger = logging.getLogger(__name__)


@check_output(DataFrameSchema(columns={}, unique=["id"]), df_name="nodes")
@check_output(DataFrameSchema(columns={}, unique=["object", "predicate", "subject"]), df_name="edges")
@inject_object()
def sample_knowledge_graph(
    sampler: Sampler,
    knowledge_graph_nodes: ps.DataFrame,
    knowledge_graph_edges: ps.DataFrame,
    ground_truth_edges: ps.DataFrame,
) -> Dict[str, ps.DataFrame]:
    return sampler.sample(knowledge_graph_nodes, knowledge_graph_edges, ground_truth_edges)
