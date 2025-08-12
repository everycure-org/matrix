import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
from click.testing import CliRunner
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from matrix.cli_commands.experiment import _submit, build_argo_template, get_run_name, run, save_argo_template
from matrix.kedro4argo_node import ArgoResourceConfig


@pytest.fixture
def mock_run_subprocess():
    with patch("matrix.cli_commands.experiment.run_subprocess") as mock:
        mock.return_value = MagicMock(stdout='{"metadata": {"name": "mocked-job-name"}}')
        yield mock


@pytest.fixture
def mock_namespace_exists():
    with patch("matrix.cli_commands.experiment.namespace_exists") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_apply():
    with patch("matrix.cli_commands.experiment.apply") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_submit_workflow():
    with patch("matrix.cli_commands.experiment.submit_workflow_from_file") as mock:
        mock.return_value = "dummy_workflow_name"
        yield mock


@pytest.fixture
def mock_can_talk_to_kubernetes():
    with patch("matrix.cli_commands.experiment.can_talk_to_kubernetes") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_argo_template_lint():
    with patch("matrix.cli_commands.experiment.argo_template_lint") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture(scope="function")
def mock_pipelines():
    pipeline_dict = {
        "__default__": MagicMock(),
        "mock_pipeline": MagicMock(),
        "mock_pipeline2": MagicMock(),
        "mock_pipeline3": MagicMock(),
    }

    with patch("matrix.cli_commands.experiment.kedro_pipelines", new=pipeline_dict) as mock:
        yield mock


@pytest.fixture
def mock_submit_internal():
    with patch("matrix.cli_commands.experiment._submit") as mock:
        yield mock


@pytest.fixture(scope="function")
def mock_multiple_pipelines():
    pipeline_dict = {
        "mock_pipeline": MagicMock(),
        "mock_pipeline2": MagicMock(),
        "mock_pipeline3": MagicMock(),
    }

    with patch("matrix.cli_commands.experiment.kedro_pipelines", new=pipeline_dict) as mock:
        yield mock


@patch("matrix.cli_commands.experiment.generate_argo_config")
def test_build_argo_template(mock_generate_argo_config: None) -> None:
    build_argo_template(
        "image",
        "test_run",
        "testuser",
        "test_namespace",
        {"test": MagicMock()},
        ArgoResourceConfig(),
        "cloud",
        is_test=True,
        mlflow_experiment_id=1,
        mlflow_url="https://mlflow.platform.dev.everycure.org/",
    )
    mock_generate_argo_config.assert_called_once()


@pytest.fixture()
def temporary_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_save_argo_template_creates_file(temporary_directory: Path) -> None:
    argo_template = "test template content"
    result = save_argo_template(argo_template, temporary_directory)

    assert Path(result).exists()
    assert Path(result).is_file()


def test_save_argo_template_content(temporary_directory: Path) -> None:
    argo_template = "test template content"

    file_path = save_argo_template(argo_template, temporary_directory)

    with open(file_path, "r") as f:
        content = f.read()

    assert content == argo_template


def test_save_argo_template_returns_string(temporary_directory: Path) -> None:
    argo_template = "test template content"

    result = save_argo_template(argo_template, temporary_directory)

    assert isinstance(result, str)


@pytest.mark.parametrize(
    "input_name,expected_name",
    [
        ("custom_name", "custom-name"),
        ("custom-name", "custom-name"),
        ("custom@name!", "custom-name"),
    ],
)
def test_get_run_name_with_input(input_name: str, expected_name: str) -> None:
    assert expected_name in get_run_name(input_name)


@pytest.mark.skip(reason="Investigate why click is not correctly throwing up exceptions")
def test_pipeline_not_found(mock_multiple_pipelines):
    with pytest.raises(click.ClickException):
        # Given a CLI runner instance
        runner = CliRunner()

        # When invoking with non existing pipeline
        runner.invoke(run, ["--username", "testuser", "--run-name", "test-run", "--pipeline", "not_exists"])


@pytest.mark.parametrize("pipeline_for_execution", ["__default__", "test_pipeline"])
def test_workflow_submission(
    mock_run_subprocess: None,
    mock_argo_template_lint,
    mock_namespace_exists,
    mock_apply,
    mock_can_talk_to_kubernetes,
    mock_submit_workflow,
    temporary_directory: Path,
    pipeline_for_execution: str,
) -> None:
    def dummy_func(*args):
        """Dummy function for testing purposes."""
        return args

    pipeline_obj = Pipeline(
        nodes=[Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")]
    )
    pipeline_obj.name = pipeline_for_execution

    _submit(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        release_version="test_release",
        pipeline_obj=pipeline_obj,
        verbose=True,
        dry_run=False,
        template_directory=temporary_directory,
        mlflow_experiment_id=1,
        allow_interactions=False,
        environment="cloud",
    )

    yaml_file = temporary_directory / "argo-workflow-template.yml"
    assert yaml_file.is_file(), f"Expected {yaml_file} to be a file"

    with open(yaml_file, "r") as f:
        content = yaml.safe_load(f)

    assert isinstance(content, dict), "Parsed YAML content should be a dictionary"

    templates = content.get("spec", {}).get("templates", [])
    pipeline_templates = [t for t in templates if "dag" in t]

    assert len(pipeline_templates) == 1, "Expected one pipeline template (test and cloud)"

    pipeline_names = [t["name"] for t in pipeline_templates]
    assert "pipeline" in pipeline_names, "Expected 'pipeline' pipeline to be present"

    # Additional checks
    assert content["metadata"]["name"] == "test-run", "Expected 'test-run' as the workflow name"
    assert content["metadata"]["namespace"] == "test_namespace", "Expected 'test_namespace' as the namespace"

    # Check for the presence of tasks in each pipeline
    for pipeline in pipeline_templates:
        tasks = pipeline.get("dag", {}).get("tasks", [])
        assert len(tasks) > 0, f"Expected at least one task in the {pipeline['name']} pipeline"

    mock_argo_template_lint.assert_called_with(str(yaml_file), verbose=True)
    mock_can_talk_to_kubernetes.assert_called_once()
    mock_namespace_exists.assert_called_with("test_namespace")
    mock_apply.assert_called_with("test_namespace", str(yaml_file), verbose=True)
    mock_submit_workflow.assert_called_with("test-run", "test_namespace", verbose=True)
