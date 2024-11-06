from kedro.pipeline import pipeline, Pipeline
from kedro.pipeline.node import Node
import logging
import pandas as pd

from matrix.kedro_extension import KubernetesNode

from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def get_parametrized_node(node_class: Node) -> Node:
    def dummy_func(x: int) -> int:
        return 2 * x

    return node_class(
        func=dummy_func,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
        name="dummy_node",
        tags=["dummy_tag"],
        confirms=["dummy_confirm"],
        namespace="dummy_namespace",
    )


def get_parametrized_pipeline(node_class: Node) -> Pipeline:
    """This fixture returns a pipeline, parametrized with node class."""

    def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
        """Splits data into features and targets training and test sets.

        Args:
            data: Data containing features and target.
            parameters: Parameters defined in parameters/data_science.yml.
        Returns:
            Split data.
        """
        X = data[parameters["features"]]
        y = data["price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
        )
        return X_train, X_test, y_train, y_test

    def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        """Trains the linear regression model.

        Args:
            X_train: Training data of independent features.
            y_train: Training data for price.

        Returns:
            Trained model.
        """
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        return regressor

    def evaluate_model(regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Calculates and logs the coefficient of determination.

        Args:
            regressor: Trained model.
            X_test: Testing data of independent features.
            y_test: Testing data for price.
        """
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        me = max_error(y_test, y_pred)
        logger = logging.getLogger(__name__)
        logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
        return {"r2_score": score, "mae": mae, "max_error": me}

    return pipeline(
        [
            node_class(
                func=split_data,
                inputs=["model_input_table@pandas", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node_class(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model_node",
            ),
            node_class(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )


def test_parametrized_node():
    normal_node = get_parametrized_node(Node)
    assert normal_node.func(2) == 4

    k8s_node = get_parametrized_node(KubernetesNode)
    assert k8s_node.func(2) == 4
