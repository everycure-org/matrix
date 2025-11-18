import pandas as pd
import pandas.testing as pdt
import pyspark.sql as ps
import pytest
import yaml
from kedro.io.core import DatasetError
from matrix_gcp_datasets.catalog import DataCatalogDataset, DatasetModel, best_match
from matrix_gcp_datasets.storage import GitStorageService, LocalStorageService
from pyspark.sql import SparkSession
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def local_storage_service(tmpdir, monkeypatch):
    storage = LocalStorageService(str(tmpdir))
    monkeypatch.setattr(GitStorageService, "get_instance", staticmethod(lambda: storage))
    return storage


@pytest.fixture
def patch_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 42, 999_999_999, 3_000_000_000, 8_000_000_000], "b": ["x", "y", "z", "w", "v"]})


@pytest.fixture
def major_df() -> pd.DataFrame:
    return pd.DataFrame({"c": [1, 42, 999_999_999, 3_000_000_000, 8_000_000_000], "d": ["x", "y", "z", "w", "v"]})


@pytest.fixture
def dummy_dataset_patch(tmpdir, patch_df):
    # Create CSV file
    csv_file = tmpdir.join("bucket", "path", "to", "minor.csv")
    csv_file.dirpath().ensure(dir=True)
    patch_df.to_csv(str(csv_file), index=False)

    # Create yaml
    dataset_yaml = {
        "name": "dummy",
        "version": "0.0.1",
        "location": {"type": "gcs", "uri": str(csv_file), "format": "csv"},
        "owner": {"name": "svc", "email": "svc@example.com"},
    }

    yaml_dir = tmpdir / "datasets" / "dummy" / "0.0.1" / "dataset.yaml"
    yaml_dir.dirpath().ensure(dir=True)
    yaml_dir.write_text(yaml.safe_dump(dataset_yaml), encoding="utf-8")


@pytest.fixture
def dummy_dataset_major(tmpdir, major_df):
    # Create CSV file
    csv_file = tmpdir.join("bucket", "path", "to", "major.csv")
    csv_file.dirpath().ensure(dir=True)
    major_df.to_csv(str(csv_file), index=False)

    # Create yaml
    dataset_yaml = {
        "name": "dummy",
        "version": "0.0.1",
        "location": {"type": "gcs", "uri": str(csv_file), "format": "csv"},
        "owner": {"name": "svc", "email": "svc@example.com"},
    }

    yaml_dir = tmpdir / "datasets" / "dummy" / "1.0.1" / "dataset.yaml"
    yaml_dir.dirpath().ensure(dir=True)
    yaml_dir.write_text(yaml.safe_dump(dataset_yaml), encoding="utf-8")


@pytest.mark.parametrize(
    "versions, pattern, expected_match, expected_is_latest",
    [
        # Exact version match
        (["1.0.0", "1.2.0", "2.0.0"], "1.2.0", "1.2.0", False),
        # Range match (should pick highest satisfying version)
        (["1.0.0", "1.4.0", "2.0.0"], "^1.0.0", "1.4.0", False),
        # 3️Latest version match (pattern allows all)
        (["1.0.0", "1.5.0", "2.0.0"], "*", "2.0.0", True),
    ],
)
def test_best_match(versions, pattern, expected_match, expected_is_latest):
    match, is_latest = best_match(versions, pattern)
    assert match == expected_match
    assert is_latest == expected_is_latest


def test_load_versions_pandas(local_storage_service, dummy_dataset_patch, patch_df):
    # Given an instance of the catalog dataset
    dataset = DataCatalogDataset(
        dataset="dummy", engine="pandas", load_args={"version": "~0.0.0", "assert_latest": False}
    )

    # Then versions loaded correctly
    assert dataset.versions == ["0.0.1"]
    result = dataset.load()
    assert isinstance(result, pd.DataFrame)
    pdt.assert_frame_equal(dataset.load(), patch_df)


def test_load_versions_spark(local_storage_service, dummy_dataset_patch, patch_df):
    # Given an instance of the catalog dataset
    dataset = DataCatalogDataset(
        dataset="dummy", engine="spark", load_args={"version": "~0.0.0", "assert_latest": False, "inferSchema": True}
    )

    # Then versions loaded correctly
    assert dataset.versions == ["0.0.1"]
    result = dataset.load()
    assert isinstance(result, ps.DataFrame)

    # Assert output is correct Spark dataframe
    spark = SparkSession.builder.getOrCreate()
    patch_spark_df = spark.createDataFrame(patch_df)
    assertDataFrameEqual(dataset.load(), patch_spark_df)


def test_load_versions_assert_latest(local_storage_service, dummy_dataset_patch, dummy_dataset_major, monkeypatch):
    with pytest.raises(DatasetError):
        # Given an instance of the catalog dataset
        dataset = DataCatalogDataset(
            dataset="dummy", engine="pandas", load_args={"version": "~0.0.1", "assert_latest": True}
        )

        # Then versions loaded correctly
        assert sorted(dataset.versions) == sorted(["1.0.1", "0.0.1"])

        # And dataset error thrown on load
        dataset.load()


def test_save(tmpdir, local_storage_service, patch_df):
    # Given an instance of the catalog dataset
    dataset = DataCatalogDataset(
        dataset="dummy",
        engine="pandas",
        save_args={
            "version": "0.1.1",
            "filepath": f"{tmpdir}/patch.csv",
            "format": "csv",
            "name": "foo",
            "email": "bar",
        },
    )

    # When saving the dataset
    dataset.save(patch_df)

    # Then metadata should be present, and data stored correctly
    assert local_storage_service.exists("datasets/dummy/0.1.1/dataset.yaml")
    catalog_entry = DatasetModel.model_validate(
        yaml.safe_load(local_storage_service.get("datasets/dummy/0.1.1/dataset.yaml"))
    )
    assert catalog_entry.schema == {"a": "int", "b": "string"}
    pdt.assert_frame_equal(pd.read_csv(catalog_entry.location.uri), patch_df)
