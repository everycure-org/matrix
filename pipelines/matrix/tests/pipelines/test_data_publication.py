import pandas as pd
from matrix.pipelines.data_publication.pipeline import (
    _drop_disease_hf_columns,
    create_pipeline,
)


class TestDropDiseaseHfColumns:
    """Tests for the _drop_disease_hf_columns function."""

    def test_drops_specified_columns(self):
        df = pd.DataFrame(
            {
                "id": ["MONDO:0000001"],
                "name": ["test disease"],
                "unmet_medical_need": [0.5],
                "is_psychiatric_disease": [True],
                "is_malignant_cancer": [False],
                "is_benign_tumour": [False],
                "is_infectious_disease": [True],
                "is_glucose_dysfunction": [False],
            }
        )
        result = _drop_disease_hf_columns(df)
        assert list(result.columns) == ["id", "name"]

    def test_ignores_missing_columns(self):
        df = pd.DataFrame(
            {
                "id": ["MONDO:0000001"],
                "name": ["test disease"],
                "is_malignant_cancer": [False],
            }
        )
        result = _drop_disease_hf_columns(df)
        assert list(result.columns) == ["id", "name"]

    def test_preserves_all_rows(self):
        df = pd.DataFrame(
            {
                "id": ["A", "B", "C"],
                "unmet_medical_need": [0.1, 0.2, 0.3],
            }
        )
        result = _drop_disease_hf_columns(df)
        assert len(result) == 3
        assert list(result["id"]) == ["A", "B", "C"]


class TestDataPublicationPipeline:
    """Tests for the data_publication pipeline structure."""

    def test_pipeline_has_expected_nodes(self):
        pipeline = create_pipeline()
        node_names = {n.name for n in pipeline.nodes}
        assert node_names == {
            "publish_kg_edges_node",
            "publish_kg_nodes_node",
            "publish_drug_list_node",
            "publish_disease_list_node",
        }

    def test_drug_list_node_is_passthrough(self):
        pipeline = create_pipeline()
        drug_node = next(n for n in pipeline.nodes if n.name == "publish_drug_list_node")
        assert drug_node.func("test_input") == "test_input"

    def test_disease_list_node_uses_column_drop(self):
        pipeline = create_pipeline()
        disease_node = next(n for n in pipeline.nodes if n.name == "publish_disease_list_node")
        assert disease_node.func is _drop_disease_hf_columns
