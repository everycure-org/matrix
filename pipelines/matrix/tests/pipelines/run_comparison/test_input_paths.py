from matrix.pipelines.run_comparison.input_paths import InputPathSingleFold, InputPathsModellingRun, InputPathsMultiFold


def test_input_path_single_fold():
    # Given a model name, file path, score column and file format
    name = "name"
    file_path = "filepath"
    score_col_name = "score"
    file_format = "csv"

    # When the InputPathSingleFold class is initialized
    input_paths_instance = InputPathSingleFold(name, file_path, score_col_name, file_format=file_format)

    # Then an instance of InputPathsMultiFold is returned with the correct attributes
    assert isinstance(input_paths_instance, InputPathsMultiFold)
    assert input_paths_instance.name == name
    assert input_paths_instance.file_paths_list == [file_path]
    assert input_paths_instance.score_col_name == score_col_name
    assert input_paths_instance.file_format == file_format


def test_input_path_modelling_run():
    # Given a model name, dat release, run name and number of folds
    name = "name"
    data_release = "data_release"
    run_name = "run_name"
    num_folds = 3

    # When the InputPathModellingRun class is initialized with and without is_transformed
    input_paths_instance_transformed = InputPathsModellingRun(
        name, data_release, run_name, num_folds, is_transformed=True
    )
    input_paths_instance_not_transformed = InputPathsModellingRun(
        name, data_release, run_name, num_folds, is_transformed=False
    )

    # Then an instance of InputPathsMultiFold is returned with the correct attributes
    assert isinstance(input_paths_instance_transformed, InputPathsMultiFold)
    assert isinstance(input_paths_instance_not_transformed, InputPathsMultiFold)
    assert input_paths_instance_transformed.name == name
    assert input_paths_instance_not_transformed.name == name
    assert input_paths_instance_transformed.file_paths_list == [
        f"gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/{data_release}/runs/{run_name}/datasets/matrix_transformations/fold_{fold}/transformed_matrix"
        for fold in range(num_folds)
    ]
    assert input_paths_instance_not_transformed.file_paths_list == [
        f"gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/{data_release}/runs/{run_name}/datasets/model_output/fold_{fold}/matrix_predictions"
        for fold in range(num_folds)
    ]
    assert input_paths_instance_transformed.score_col_name == "transformed_treat_score"
    assert input_paths_instance_not_transformed.score_col_name == "treat score"
    assert input_paths_instance_transformed.file_format == "parquet"
    assert input_paths_instance_not_transformed.file_format == "parquet"
