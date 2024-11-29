from matrix.pipelines.gpu_validate.pipeline import check_gpu_availability


def test_gpu_model_training():
    """
    Given a GPU is available
    When running the check_gpu_availability function
    Then it should successfully train a model and return valid results
    """
    # When
    results = check_gpu_availability()

    # Then
    assert "gpu" in results.columns
    assert "cuda_device" in results.columns
    assert "final_loss" in results.columns
    assert "sample_predictions" in results.columns

    # Verify loss is a reasonable value (should be less than initial random predictions)
    assert 0 < results["final_loss"].iloc[0] < 1.0

    # Verify predictions are in the expected format and range
    predictions = results["sample_predictions"].iloc[0]
    assert isinstance(predictions, list)
    assert len(predictions) == 5  # We requested 5 predictions
