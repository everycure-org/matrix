from matrix.pipeline_registry import register_pipelines
from matrix.pipelines.data_release import last_node_name

pipeline_registry = register_pipelines()


class TestDataReleasePipeline:
    def test_last_node_is_sentinel_kg_release_patch(self):
        """
        This tests that nothing runs AFTER the sentinel node when running the KG release patch pipeline
        """
        pipeline = pipeline_registry["kg_release_patch"]

        node_dependencies = pipeline.node_dependencies
        all_dependencies = set()
        for node_deps in node_dependencies.values():
            all_dependencies.update(node.name for node in node_deps)

        terminal_nodes = [node.name for node in pipeline.nodes if node.name not in all_dependencies]

        assert len(terminal_nodes) == 1, f"Expected 1 terminal node, found {len(terminal_nodes)}: {terminal_nodes}"
        assert terminal_nodes[0] == last_node_name, f"Last node should be {last_node_name}, found {terminal_nodes[0]}"

    def test_last_node_is_sentinel_kg_release(self):
        """
        This tests that nothing runs AFTER the sentinel node when running the KG release pipeline
        """
        pipeline = pipeline_registry["kg_release"]

        node_dependencies = pipeline.node_dependencies
        all_dependencies = set()
        for node_deps in node_dependencies.values():
            all_dependencies.update(node.name for node in node_deps)

        terminal_nodes = [node.name for node in pipeline.nodes if node.name not in all_dependencies]

        assert len(terminal_nodes) == 1, f"Expected 1 terminal node, found {len(terminal_nodes)}: {terminal_nodes}"
        assert terminal_nodes[0] == last_node_name, f"Last node should be {last_node_name}, found {terminal_nodes[0]}"

    def test_all_nodes_are_upstream_of_last_node_kg_release(self):
        """
        Tests that everything runs BEFORE the last_node
        It's not the same as the function above - test_last_node_is_sentinel_kg_release
        because there could be disconnected node that runs in parallel but never feeds into the sentinel
        """
        pipeline = pipeline_registry["kg_release"]
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

    def test_all_nodes_are_upstream_of_last_node_kg_release_patch(self):
        """
        Tests that everything runs BEFORE the last_node
        It's not the same as the function above - test_last_node_is_sentinel_kg_release_patch
        because there could be disconnected node that runs in parallel but never feeds into the sentinel
        """
        pipeline = pipeline_registry["kg_release_patch"]
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
