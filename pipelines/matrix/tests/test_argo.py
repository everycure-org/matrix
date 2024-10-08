from pathlib import Path
from typing import Dict, Any
from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline
import pytest
import yaml

from matrix.argo import clean_name, fuse, FusedNode, generate_argo_config


def dummy_fn(*args):
    return "dummy"


def test_no_nodes_fused_when_no_fuse_options():
    pipeline_with_no_fusing_options = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_c",
                name="first",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_1", "dataset_2"],  # inputs are different than outputs of previous node
                outputs="dataset_3",
                name="second",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused = fuse(pipeline_with_no_fusing_options)

    assert len(fused) == len(
        pipeline_with_no_fusing_options.nodes
    ), "No nodes should be fused when no fuse options are provided"


def test_simple_fusing():
    pipeline_where_first_node_is_input_for_second = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_1@pandas",
            ),
            Node(
                func=dummy_fn,
                inputs=[
                    "dataset_1@spark",
                ],
                outputs="dataset_2",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused = fuse(pipeline_where_first_node_is_input_for_second)

    assert len(fused) == 1, "Only one node should be fused"
    assert fused[0].name == "dummy", "Fused node should have name 'dummy'"
    assert fused[0].outputs == set(
        ["dataset_1", "dataset_2"]
    ), "Fused node should have outputs 'dataset_1' and 'dataset_2'"
    assert len(fused[0]._parents) == 0, "Fused node should have no parents"


def test_no_multiple_parents_no_fusing():
    pipeline_one2many_fusing_possible = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_1",
                name="first_node",
            ),
            Node(
                func=dummy_fn,
                inputs=[
                    "dataset_c",
                ],
                outputs="dataset_2",
                name="second_node",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_1", "dataset_2"],
                outputs="dataset_3",
                name="child_node",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused = fuse(pipeline_one2many_fusing_possible)

    assert len(fused) == len(
        pipeline_one2many_fusing_possible.nodes
    ), "No fusing has been performed, as child node can be fused to different parents."


def test_fusing_multiple_parents():
    pipeline_with_multiple_parents = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs=["dataset_1"],
                name="first_node",
            ),
            Node(
                func=dummy_fn,
                inputs=[
                    "dataset_c",
                ],
                outputs="dataset_2",
                name="second_node",
            ),
            Node(
                func=dummy_fn,
                inputs=None,
                outputs="dataset_3",
                name="third_node",
            ),
            Node(
                func=dummy_fn,
                inputs=[
                    "dataset_1",
                    "dataset_2",
                ],
                outputs="dataset_4",
                name="child_node",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_3", "dataset_4"],
                outputs="dataset_5",
                name="grandchild_node",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_5"],
                outputs="dataset_6",
                name="grandgrandchild_node",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused = fuse(pipeline_with_multiple_parents)

    assert len(fused) == 4, "Fusing of child and grandchild node, ensure correct naming"
    assert fused[3].name == "dummy", "Fused node should have name 'dummy'"
    assert (
        fused[3].nodes == "child_node,grandchild_node,grandgrandchild_node"
    ), "Fused node should have nodes 'child_node,grandchild_node,grandgrandchild_node'"
    assert fused[3].outputs == set(
        ["dataset_4", "dataset_5", "dataset_6"]
    ), "Fused node should have outputs 'dataset_4', 'dataset_5' and 'dataset_6'"
    assert set([parent.name for parent in fused[3]._parents]) == set(
        ["first_node", "second_node", "third_node"]
    ), "Fused node should have parents 'first_node', 'second_node' and 'third_node'"


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("node_name_1", "node-name-1"),  # Basic case with underscore
        ("node@name!", "node-name"),  # Special characters
        ("_node_name_", "node-name"),  # Leading and trailing underscores
        ("-node_name-", "node-name"),  # Leading and trailing dashes
        ("@@node!!", "node"),  # Leading and trailing special characters
        ("", ""),  # Empty string
        ("clean-name", "clean-name"),  # Already clean name
        ("node__name---test", "node-name-test"),  # Multiple consecutive special characters
        ("name!!!node$$$name", "name-node-name"),  # Complex case with multiple special characters
    ],
)
def test_clean_name(input_name: str, expected: str) -> None:
    """Test clean_name function with various input cases."""
    assert clean_name(input_name) == expected


def dummy_func(*args) -> str:
    return "dummy"


@pytest.fixture()
def simple_node():
    return Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")


@pytest.fixture()
def fused_node():
    return FusedNode(depth=0)


def test_fused_node_initialization(fused_node: FusedNode) -> None:
    assert fused_node.depth == 0
    assert fused_node._nodes == []
    assert fused_node._parents == set()
    assert fused_node._inputs == []


def test_add_node(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    assert len(fused_node._nodes) == 1
    assert fused_node._nodes[0] == simple_node


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_add_parents(fused_node: FusedNode):
    parent1 = FusedNode(depth=1)
    parent2 = FusedNode(depth=1)
    fused_node.add_parents([parent1, parent2])
    assert len(fused_node._parents) == 2
    assert parent1 in fused_node._parents
    assert parent2 in fused_node._parents


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_fuses_with(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["argowf.fuse", "argowf.fuse-group.test"]

    fusable_node = Node(
        func=dummy_func,
        inputs=["dataset_c"],
        outputs="dataset_d",
        name="fusable_node",
        tags=["argowf.fuse", "argowf.fuse-group.test"],
    )

    assert fused_node.fuses_with(fusable_node)


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_not_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    non_fusable_node = Node(func=dummy_func, inputs=["dataset_x"], outputs="dataset_y", name="non_fusable_node")

    assert not fused_node.fuses_with(non_fusable_node)


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_is_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["argowf.fuse"]
    assert fused_node.is_fusable


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_not_is_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    assert not fused_node.is_fusable


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_fuse_group(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["argowf.fuse-group.test_group"]
    assert fused_node.fuse_group == "test_group"


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_nodes_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node.add_node(Node(func=dummy_func, name="second_node"))
    assert fused_node.nodes == "simple_node,second_node"


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_outputs_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node.add_node(Node(func=dummy_func, outputs="dataset_d"))
    assert fused_node.outputs == {"dataset_c", "dataset_d"}


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_tags_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["tag1", "tag2"]
    fused_node.add_node(Node(func=dummy_func, tags=["tag2", "tag3"]))
    assert fused_node.tags == {"tag1", "tag2", "tag3"}


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_name_property_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["argowf.fuse", "argowf.fuse-group.test_group"]
    fused_node.add_node(Node(func=dummy_func, name="second_node"))
    assert fused_node.name == "test_group"


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_name_property_not_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    assert fused_node.name == "simple_node"


def test_get_fuse_group() -> None:
    tags = ["argowf.fuse-group.test_group", "other_tag"]
    assert FusedNode.get_fuse_group(tags) == "test_group"


def test_get_fuse_group_no_group() -> None:
    tags = ["other_tag"]
    assert FusedNode.get_fuse_group(tags) is None


def test_clean_dependencies() -> None:
    elements = ["dataset_a@pandas", "params:some_param", "dataset_b"]
    cleaned = FusedNode.clean_dependencies(elements)
    assert cleaned == ["dataset_a", "dataset_b"]


@pytest.fixture()
def expected_argo_config():
    """Fixture to provide expected Argo YAML output for comparison."""
    return yaml.safe_load(
        """
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  namespace: test_namespace
  name: test_run
spec:
  workflowMetadata:
    labels:
      run: '{{ workflow.parameters.run_name }}'
      username: "test_user"
  entrypoint: dag
  ttlStrategy:
    secondsAfterSuccess: 5
  arguments:
    parameters:
      - name: image
        value: "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
      - name: image_tag
        value: "test_tag"
      - name: run_name
        value: "test_tag"
      - name: neo4j_host
        value: "bolt://neo4j.neo4j.svc.cluster.local:7687"
      - name: mlflow_endpoint
        value: "http://mlflow-tracking.mlflow.svc.cluster.local:80"
      - name: openai_endpoint
        value: "https://api.openai.com/v1"
      - name: env
        value: "cloud"
  templates:
  - name: kedro
    backoff:
      duration: "1"      # Must be a string. Default unit is seconds. Could also be a Duration, e.g.: "2m", "6h", "1d"
      factor: 2
      maxDuration: "1m"  # Must be a string. Default unit is seconds. Could also be a Duration, e.g.: "2m", "6h", "1d"
    affinity:
      nodeAntiAffinity: {}
    metadata:
      labels:
        app: kedro-argo
    inputs:
      parameters:
      - name: kedro_nodes
      - name: pipeline
    container:
      imagePullPolicy: Always
      image:  "{{workflow.parameters.image}}:{{workflow.parameters.image_tag}}" 
      resources: # limit the resources
        requests:
          memory: 64Gi
          cpu: 4
        limits:
          memory: 64Gi
          cpu: 16
      env:
        - name: WORKFLOW_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['workflows.argoproj.io/workflow']
        - name: RUN_NAME
          value:  "{{workflow.parameters.run_name}}" 
        - name: NEO4J_HOST
          value:  "{{workflow.parameters.neo4j_host}}" 
        - name: MLFLOW_ENDPOINT
          value:  "{{workflow.parameters.mlflow_endpoint}}" 
        - name: NEO4J_USER
          valueFrom:
            secretKeyRef:
              name: matrix-secrets
              key: NEO4J_USER
        - name: NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: matrix-secrets
              key: NEO4J_PASSWORD
        - name: OPENAI_ENDPOINT
          value:  "{{workflow.parameters.openai_endpoint}}" 
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: matrix-secrets
              key: OPENAI_API_KEY
        - name: GCP_PROJECT_ID
          valueFrom:
            configMapKeyRef:
              name: matrix-config
              key: GCP_PROJECT_ID
        - name: GCP_BUCKET
          valueFrom:
            configMapKeyRef:
              name: matrix-config
              key: GCP_BUCKET
      command: [kedro]
      args: ["run", "-p", "{{inputs.parameters.pipeline}}", "-e", "{{workflow.parameters.env}}", "-n", "{{inputs.parameters.kedro_nodes}}"]

  - name: neo4j
    inputs:
      parameters:
      - name: kedro_nodes
      - name: pipeline
    container:
      resources: # limit the resources
        requests:
          memory: 120Gi
          cpu: 4
        limits:
          memory: 120Gi
          cpu: 16
      env:
        - name: WORKFLOW_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['workflows.argoproj.io/workflow']
        - name: RUN_NAME
          value:  "{{workflow.parameters.run_name}}" 
        - name: MLFLOW_ENDPOINT
          value:  "{{workflow.parameters.mlflow_endpoint}}" 
        - name: NEO4J_USER
          value: "neo4j"
        - name: NEO4J_PASSWORD
          value: "admin"
        - name: GCP_PROJECT_ID
          valueFrom:
            configMapKeyRef:
              name: matrix-config
              key: GCP_PROJECT_ID
        - name: GCP_BUCKET
          valueFrom:
            configMapKeyRef:
              name: matrix-config
              key: GCP_BUCKET
        - name: OPENAI_API_KEY
          value: "foo"
        - name: OPENAI_ENDPOINT
          value:  "{{workflow.parameters.openai_endpoint}}" 
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: matrix-secrets
              key: OPENAI_API_KEY
      imagePullPolicy: Always
      image:  "{{workflow.parameters.image}}:{{workflow.parameters.image_tag}}"       
      command: ["/bin/sh", "-c"]
      args:
        - |
          echo "Waiting for Neo4j to be ready..."
          until curl -s http://localhost:7474/ready; do
            echo "Waiting..."
            sleep 5
          done
          echo "Neo4j is ready. Starting main application..."
          kedro run -p "{{inputs.parameters.pipeline}}" -e "{{workflow.parameters.env}}" -n "{{inputs.parameters.kedro_nodes}}"    
    sidecars:
      - name: neo4j
        image: neo4j:5.21.0-enterprise
        env:
        - name: NEO4J_AUTH
          value: "neo4j/admin"
        - name: NEO4J_apoc_export_file_enabled
          value: "true"
        - name: NEO4J_apoc_import_file_enabled
          value: "true"
        - name: NEO4J_apoc_import_file_use__neo4j__config
          value: "true"
        - name: NEO4J_PLUGINS
          value: '["apoc", "graph-data-science", "apoc-extended"]'
        - name: NEO4J_dbms_security_auth__minimum__password__length
          value: "4"
        - name: NEO4J_dbms_security_procedures_whitelist
          value: "gds.*, apoc.*"
        - name: NEO4J_dbms_security_procedures_unrestricted
          value: "gds.*, apoc.*"
        - name: NEO4J_db_logs_query_enabled
          value: "OFF"
        - name: NEO4J_ACCEPT_LICENSE_AGREEMENT
          value: "yes"
  - name: test
    dag:
      tasks:
      - name: simple-node
        template: kedro
        arguments:
          parameters:
          - name: pipeline
            value: test
          - name: kedro_nodes
            value: simple_node
        """
    )


def test_generate_argo_config(expected_argo_config: Dict[str, Any], matrix_root: Path) -> None:
    image_name = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
    run_name = "test_run"
    image_tag = "test_tag"
    namespace = "test_namespace"
    username = "test_user"
    pipelines = {
        "test": Pipeline(
            nodes=[Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")]
        )
    }

    argo_config = generate_argo_config(
        image=image_name,
        run_name=run_name,
        image_tag=image_tag,
        namespace=namespace,
        username=username,
        pipelines=pipelines,
        project_path=matrix_root,
    )

    assert argo_config is not None

    parsed_config = yaml.safe_load(argo_config)

    assert isinstance(parsed_config, dict), "Parsed config should be a dictionary"
    # Verify apiVersion
    assert parsed_config["apiVersion"] == expected_argo_config["apiVersion"], "Config should have 'apiVersion'"

    # Verify kind
    assert parsed_config["kind"] == expected_argo_config["kind"], "Config should have 'kind'"

    # Verify metadata
    assert (
        parsed_config["metadata"]["namespace"] == expected_argo_config["metadata"]["namespace"]
    ), "Config should have correct 'namespace'"
    assert (
        parsed_config["metadata"]["name"] == expected_argo_config["metadata"]["name"]
    ), "Config should have correct 'name'"

    # Verify spec
    spec = parsed_config["spec"]
    expected_spec = expected_argo_config["spec"]

    # Verify workflowMetadata
    assert (
        spec["workflowMetadata"]["labels"]["run"] == expected_spec["workflowMetadata"]["labels"]["run"]
    ), "Config should have correct 'run' label"
    assert (
        spec["workflowMetadata"]["labels"]["username"] == expected_spec["workflowMetadata"]["labels"]["username"]
    ), "Config should have correct 'username' label"

    # Verify entrypoint and ttlStrategy
    assert spec["entrypoint"] == expected_spec["entrypoint"], "Config should have correct 'entrypoint'"
    assert (
        spec["ttlStrategy"]["secondsAfterSuccess"] == expected_spec["ttlStrategy"]["secondsAfterSuccess"]
    ), "Config should have correct 'ttlStrategy'"

    # Verify arguments
    parameters_actual = spec["arguments"]["parameters"]
    parameters_expected = expected_spec["arguments"]["parameters"]
    assert len(parameters_actual) == len(parameters_expected), "Config should have correct number of parameters"

    for parameter_expected, parameter_actual in zip(parameters_expected, parameters_actual):
        assert (
            parameter_expected["name"] == parameter_actual["name"]
        ), f"Parameter {parameter_expected['name']} not found or mismatched"
        assert (
            parameter_expected["value"] == parameter_actual["value"]
        ), f"Parameter {parameter_expected['name']} has incorrect value"

    # Verify templates
    templates = spec["templates"]
    assert len(templates) == 3, "Config should have 3 templates"

    # Verify kedro template
    kedro_template = next(t for t in templates if t["name"] == "kedro")
    assert kedro_template["backoff"]["duration"] == "1", "Kedro template should have correct backoff duration"
    assert kedro_template["backoff"]["factor"] == 2, "Kedro template should have correct backoff factor"
    assert kedro_template["backoff"]["maxDuration"] == "1m", "Kedro template should have correct max backoff duration"
    assert "nodeAntiAffinity" in kedro_template["affinity"], "Kedro template should have nodeAntiAffinity"
    assert kedro_template["metadata"]["labels"]["app"] == "kedro-argo", "Kedro template should have correct label"

    # Verify kedro container
    kedro_container = kedro_template["container"]
    assert kedro_container["imagePullPolicy"] == "Always", "Kedro container should have correct imagePullPolicy"
    assert (
        kedro_container["image"] == "{{workflow.parameters.image}}:{{workflow.parameters.image_tag}}"
    ), "Kedro container should have correct image"
    assert (
        kedro_container["resources"]["requests"]["memory"] == "64Gi"
    ), "Kedro container should have correct memory request"
    assert kedro_container["resources"]["limits"]["cpu"] == 16, "Kedro container should have correct CPU limit"

    # Verify neo4j template
    neo4j_template = next(t for t in templates if t["name"] == "neo4j")
    assert "kedro_nodes" in neo4j_template["inputs"]["parameters"], "Neo4j template should have kedro_nodes input"
    assert "pipeline" in neo4j_template["inputs"]["parameters"], "Neo4j template should have pipeline input"

    # Verify neo4j container
    neo4j_container = neo4j_template["container"]
    assert (
        neo4j_container["resources"]["requests"]["memory"] == "120Gi"
    ), "Neo4j container should have correct memory request"
    assert neo4j_container["resources"]["limits"]["cpu"] == 16, "Neo4j container should have correct CPU limit"

    # Verify neo4j sidecar
    neo4j_sidecar = neo4j_template["sidecars"][0]
    assert neo4j_sidecar["name"] == "neo4j", "Neo4j sidecar should have correct name"
    assert neo4j_sidecar["image"] == "neo4j:5.21.0-enterprise", "Neo4j sidecar should have correct image"
    assert any(
        env["name"] == "NEO4J_ACCEPT_LICENSE_AGREEMENT" and env["value"] == "yes" for env in neo4j_sidecar["env"]
    ), "Neo4j sidecar should accept license agreement"

    # Verify test template
    test_template = next(t for t in templates if t["name"] == "test")
    assert "dag" in test_template, "Test template should have a DAG"
    assert len(test_template["dag"]["tasks"]) == 1, "Test template should have one task"
    assert test_template["dag"]["tasks"][0]["name"] == "simple-node", "Test template should have correct task name"
    assert test_template["dag"]["tasks"][0]["template"] == "kedro", "Test template task should use kedro template"

    # Check if the pipeline is included in the templates
    pipeline_names = [template["name"] for template in templates]
    assert "test" in pipeline_names, "The 'test' pipeline should be included in the templates"
