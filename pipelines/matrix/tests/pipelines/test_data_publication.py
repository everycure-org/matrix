from matrix.pipelines.data_publication.pipeline import create_pipeline


class TestDataPublicationPipeline:
    """Tests for the data_publication pipeline structure."""

    def test_pipeline_has_expected_nodes(self):
        pipeline = create_pipeline()
        node_names = {n.name for n in pipeline.nodes}
        assert node_names == {
            "publish_kg_edges_node",
            "publish_kg_nodes_node",
        }
