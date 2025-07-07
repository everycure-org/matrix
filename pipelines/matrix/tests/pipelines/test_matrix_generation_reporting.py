import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matrix.pipelines.matrix_generation.reporting_plots import (
    MultiScoreHistogram,
    SingleScoreHistogram,
    SingleScoreLinePlot,
    SingleScoreScatterPlot,
)
from matrix.pipelines.matrix_generation.reporting_tables import (
    MatrixRunInfo,
    RankToScore,
    TopFrequentFlyers,
    TopPairs,
)
from pyspark.sql import SparkSession


@pytest.fixture()
def spark_session():
    """Fixture that provides a Spark session for testing."""
    return SparkSession.builder.config("spark.driver.memory", "1g").getOrCreate()


@pytest.fixture()
def sample_matrix_data_pandas():
    """Fixture that provides sample matrix data with multiple scores for testing."""
    return pd.DataFrame(
        {
            "source": ["drug_1", "drug_2", "drug_2", "drug_3"],
            "target": ["disease_1", "disease_1", "disease_2", "disease_3"],
            "score_1": [0.9, 0.7, 0.5, 0.3],
            "score_2": [0.1, 0.2, 0.3, 0.4],
            "rank": [1, 2, 3, 4],
        }
    )


@pytest.fixture()
def sample_matrix_data_spark(spark_session):
    """Fixture that provides sample matrix data with multiple scores for testing."""
    return spark_session.createDataFrame(
        [
            ("drug_1", "disease_1", 0.9, 0.1, 1),
            ("drug_2", "disease_1", 0.7, 0.2, 2),
            ("drug_2", "disease_2", 0.5, 0.3, 3),
            ("drug_3", "disease_3", 0.3, 0.4, 4),
        ],
        schema=["source", "target", "score_1", "score_2", "rank"],
    )


@pytest.fixture()
def sample_drugs_list(spark_session):
    """Fixture that provides sample drugs list with id and name"""
    return spark_session.createDataFrame(
        [("drug_1", "name_1"), ("drug_2", "name_2"), ("drug_3", "name_3")],
        schema=["id", "name"],
    )


@pytest.fixture()
def sample_diseases_list(spark_session):
    """Fixture that provides sample diseases list with id and name"""
    return spark_session.createDataFrame(
        [("disease_1", "name_1"), ("disease_2", "name_2"), ("disease_3", "name_3")],
        schema=["id", "name"],
    )


def test_plot_generator(sample_matrix_data_pandas):
    """Tests all plotting strategies."""
    # Given any plotting strategy
    generators = [
        SingleScoreHistogram(
            name="name",
            score_col="score_1",
            is_log_y_scale=True,
            figsize=(10, 6),
        ),
        MultiScoreHistogram(
            name="name",
            score_cols_lst=["score_1", "score_2"],
        ),
        SingleScoreLinePlot(
            name="name",
            score_col="score_1",
            is_log_y_scale=True,
            figsize=(10, 6),
        ),
        SingleScoreScatterPlot(
            name="name",
            score_col="score_1",
            n_sample=2,
            figsize=(10, 6),
            points_alpha=0.03,
            points_s=0.5,
        ),
    ]

    for generator in generators:
        # When generating the plot
        plot = generator.generate(sample_matrix_data_pandas)
        # Then:
        # The plot is a matplotlib figure
        assert isinstance(plot, plt.Figure)
        # The generator has the correct name
        assert generator.name == "name"


def test_matrix_run_info(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list):
    # Given the matrix run info generator
    generator = MatrixRunInfo(
        name="name", versions={"matrix": {"version": "1.0.0"}, "drugs": {"version": "2.0.0"}}, release="0.1"
    )

    # When generating the table
    table = generator.generate(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list)

    # Then:
    # The table is a pandas dataframe
    assert isinstance(table, pd.DataFrame)
    # The table has the correct columns
    assert table.columns.tolist() == ["key", "value"]
    # The table has the correct row information
    timestamp = table[table["key"] == "timestamp"]["value"].iloc[0]
    assert timestamp[:2] == "20" and len(timestamp) == 10
    assert table[table["key"] == "matrix_version"]["value"].iloc[0] == "1.0.0"
    assert table[table["key"] == "drugs_version"]["value"].iloc[0] == "2.0.0"
    assert table[table["key"] == "release"]["value"].iloc[0] == "0.1"


def test_top_pairs(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list):
    # Given the top pairs generator
    generator = TopPairs(name="name", n_reporting=2, score_col="score_1", columns_to_keep=["score_1", "score_2"])

    # When generating the table
    table = generator.generate(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list)

    # Then:
    # The table is a pandas dataframe
    assert isinstance(table, pd.DataFrame)
    # The table has the correct columns
    assert set(table.columns.tolist()) == set(
        ["drug_id", "drug_name", "disease_id", "disease_name", "score_1", "score_2"]
    )
    # The table has the correct values
    assert table["drug_id"].tolist() == ["drug_1", "drug_2"]
    assert table["disease_id"].tolist() == ["disease_1", "disease_1"]


def test_rank_to_score(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list):
    # Given the rank to score generator
    generator = RankToScore(name="name", ranks_lst=[1, 2], score_col="score_1")

    # When generating the table
    table = generator.generate(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list)

    # Then:
    # The table is a pandas dataframe
    assert isinstance(table, pd.DataFrame)
    # The table has the correct columns
    assert set(table.columns.tolist()) == set(["rank", "score_1"])
    # The table has the correct values
    assert table["rank"].tolist() == [1, 2]
    assert table["score_1"].tolist() == [0.9, 0.7]


def test_top_frequent_flyers(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list):
    # Given the top frequent flyers generator for drugs and diseases
    generator_drugs = TopFrequentFlyers(
        name="name", is_drug_mode=True, count_in_n_lst=[3], sort_by_col="mean", score_col="score_1"
    )
    generator_diseases = TopFrequentFlyers(
        name="name", is_drug_mode=False, count_in_n_lst=[3], sort_by_col="count_in_3", score_col="score_1"
    )

    # When generating the table
    table_drugs = generator_drugs.generate(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list)
    table_diseases = generator_diseases.generate(sample_matrix_data_spark, sample_drugs_list, sample_diseases_list)

    # Then:
    # The tables are pandas dataframes
    assert isinstance(table_drugs, pd.DataFrame)
    assert isinstance(table_diseases, pd.DataFrame)
    # The tables have the correct columns
    columns = {"id", "name", "mean", "max", "count_in_3"}
    assert set(table_drugs.columns.tolist()) == columns
    assert set(table_diseases.columns.tolist()) == columns
    # The tables have the correct values for drugs
    assert table_drugs["id"].tolist() == ["drug_1", "drug_2", "drug_3"]
    assert table_drugs["name"].tolist() == ["name_1", "name_2", "name_3"]
    assert table_drugs["mean"].tolist() == [0.9, 0.6, 0.3]
    assert table_drugs["max"].tolist() == [0.9, 0.7, 0.3]
    assert table_drugs["count_in_3"].tolist() == [1, 2, 0]
    # The tables have the correct values for diseases
    assert table_diseases["id"].tolist() == ["disease_1", "disease_2", "disease_3"]
    assert table_diseases["name"].tolist() == ["name_1", "name_2", "name_3"]
    assert table_diseases["mean"].tolist() == [0.8, 0.5, 0.3]
    assert table_diseases["max"].tolist() == [0.9, 0.5, 0.3]
    assert table_diseases["count_in_3"].tolist() == [2, 1, 0]
