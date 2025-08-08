from matrix.pipeline_registry import register_pipelines
from matrix.pipelines.data_release import last_node_name

pipeline_registry = register_pipelines()


class TestDataReleasePipeline:
    def test_sentinel_is_a_terminal_node_kg_release_patch(self):
        """
        Tests that the sentinel node is a terminal node in the KG release patch pipeline.
        This ensures nothing runs after the sentinel node.
        """
        pipeline = pipeline_registry["kg_release_patch_and_matrix_run"]

        # Find all terminal nodes (nodes that nothing depends on)
        node_dependencies = pipeline.node_dependencies
        all_dependencies = set()
        for node_deps in node_dependencies.values():
            all_dependencies.update(node.name for node in node_deps)

        terminal_nodes = [node.name for node in pipeline.nodes if node.name not in all_dependencies]

        # The sentinel should be a terminal node
        assert (
            last_node_name in terminal_nodes
        ), f"Sentinel node {last_node_name} should be a terminal node, but terminal nodes are: {terminal_nodes}"

    def test_sentinel_has_expected_inputs_kg_release_patch(self):
        """
        Tests that the sentinel node has inputs from the expected pipeline components in patch mode.
        This ensures the sentinel waits for critical outputs without being overly restrictive to ingest every terminal node.
        """
        pipeline = pipeline_registry["kg_release_patch_and_matrix_run"]

        sentinel_node = next(node for node in pipeline.nodes if node.name == last_node_name)
        sentinel_inputs = sentinel_node.inputs if hasattr(sentinel_node, "inputs") else []

        # We expect at least these core components
        expected_component_prefixes = [
            "data_release",
            "matrix_transformations",
            "evaluation",
        ]

        # Check that sentinel has inputs from expected components
        missing_components = []
        for prefix in expected_component_prefixes:
            if not any(prefix in input_name for input_name in sentinel_inputs):
                missing_components.append(prefix)

        assert not missing_components, f"Sentinel missing inputs from components: {missing_components}"

    def test_sentinel_is_a_terminal_node_kg_release(self):
        """
        Tests that the sentinel node is a terminal node in the KG release pipeline.
        This ensures nothing runs after the sentinel node.
        """
        pipeline = pipeline_registry["kg_release_and_matrix_run"]

        # Find all terminal nodes (nodes that nothing depends on)
        node_dependencies = pipeline.node_dependencies
        all_dependencies = set()
        for node_deps in node_dependencies.values():
            all_dependencies.update(node.name for node in node_deps)

        terminal_nodes = [node.name for node in pipeline.nodes if node.name not in all_dependencies]

        # The sentinel should be among the terminal nodes
        assert (
            last_node_name in terminal_nodes
        ), f"Sentinel node {last_node_name} should be a terminal node, but terminal nodes are: {terminal_nodes}"

    def test_sentinel_has_expected_inputs_kg_release(self):
        """
        Tests that the sentinel node has inputs from the expected pipeline components.
        This ensures the sentinel waits for critical outputs without being overly restrictive.
        """
        pipeline = pipeline_registry["kg_release_and_matrix_run"]

        sentinel_node = next(node for node in pipeline.nodes if node.name == last_node_name)
        sentinel_inputs = sentinel_node.inputs if hasattr(sentinel_node, "inputs") else []

        # For full release, we expect these core components
        expected_component_prefixes = [
            "data_release.prm.kg_edges",
            "data_release.prm.kg_nodes",
            "matrix_transformations",
            "evaluation",
        ]

        # Check that sentinel has inputs from expected components
        missing_components = []
        for prefix in expected_component_prefixes:
            if not any(prefix in input_name for input_name in sentinel_inputs):
                missing_components.append(prefix)

        assert not missing_components, f"Sentinel missing inputs from components: {missing_components}"
