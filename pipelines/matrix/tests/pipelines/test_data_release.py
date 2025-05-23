import pytest
from kedro.pipeline import Pipeline
from matrix.pipelines.data_release import last_node_name
from matrix.pipelines.data_release.pipeline import create_pipeline


class TestDataReleasePipeline:
    def test_last_node_is_terminal(self):
        """
        this tests that nothing runs AFTER the sentinel node
        """
        pipeline = create_pipeline()

        node_dependencies = pipeline.node_dependencies
        all_dependencies = set()
        for node_deps in node_dependencies.values():
            all_dependencies.update(node.name for node in node_deps)

        terminal_nodes = [node.name for node in pipeline.nodes if node.name not in all_dependencies]

        assert len(terminal_nodes) == 1, f"Expected 1 terminal node, found {len(terminal_nodes)}: {terminal_nodes}"
        assert terminal_nodes[0] == last_node_name, f"Last node should be {last_node_name}, found {terminal_nodes[0]}"

    def test_all_nodes_are_upstream_of_last_node(self):
        """
        Tests that everything runs BEFORE the last_node
        It's not the same as the function above - test_last_node_is_terminal
        because there could be disconnected node that runs in parallel but never feeds into the sentinel
        """
        pipeline = create_pipeline()
        node_dependencies = pipeline.node_dependencies

        sentinel = next(node for node in pipeline.nodes if node.name == last_node_name)

        upstream_nodes = set()
        sentinel_depends_on_nodes = list(node_dependencies[sentinel])

        while sentinel_depends_on_nodes:
            current = sentinel_depends_on_nodes.pop(0)
            if current.name not in upstream_nodes:
                upstream_nodes.add(current.name)
                sentinel_depends_on_nodes.extend(node_dependencies[current])

        other_nodes = set(node.name for node in pipeline.nodes if node.name != last_node_name)

        assert (
            upstream_nodes == other_nodes
        ), f"Last node should depend on all other nodes. Missing dependencies: {other_nodes - upstream_nodes}"

    def test_last_node_inputs_include_all_critical_outputs(self):
        """
        Given a data release pipeline
        When examining dataset flows
        Then the last_node inputs should encompass all critical outputs from data-producing nodes
        """
        # GIVEN
        pipeline = create_pipeline()

        # WHEN
        # Find the last_node and its inputs
        last_node = next(node for node in pipeline.nodes if node.name == last_node_name)
        last_node_inputs = set(last_node.inputs)

        # Collect all datasets that are outputs of data-producing nodes
        # Focus on KGX datasets which are the critical outputs
        kgx_outputs = set()
        for node in pipeline.nodes:
            if node.name != last_node_name and node.tags and "kgx" in node.tags:
                kgx_outputs.update(node.outputs)

        # THEN
        # Check if all critical KGX outputs are inputs to the last_node
        missing_inputs = [output for output in kgx_outputs if output not in last_node_inputs]
        assert not missing_inputs, f"Last node should consume all critical KGX outputs, but missing: {missing_inputs}"
