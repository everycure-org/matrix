import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Sample the Ground Truths:
            # Read from bucket and store unified Ground Truths
            # in the intermediate directory location so we can
            # access them later to merge them with the
            # rest of the sampled nodes and edges
            node(
                func=nodes.get_random_selection_of_ground_truths,
                inputs={
                    "gt_positives": "modelling.raw.ground_truth.positives",
                    "gt_negatives": "modelling.raw.ground_truth.negatives",
                },
                outputs="modelling.int.known_pairs",
                name="sample-ground_truths",
                tags=["sample"]
            ),
            # Make sample from unified nodes/edges and
            # save to the same dir for sampled_test.
            # Add the Ground Truths as the modelling pipeline
            # need the embeddings of the GTs
            node(
                func=nodes.get_random_sample_from_unified_files,
                inputs={
                    "nodes" : "integration.prm_ingest.unified_nodes",
                    "edges" : "integration.prm_ingest.unified_edges",
                    "known_pairs": "modelling.int.known_pairs",
                },
                outputs={
                    "nodes" : "integration.prm.unified_nodes",
                    "edges" : "integration.prm.unified_edges",
                },
                name="sample_unified_files",
                tags=["sample"]
            ),
        ]
    )





