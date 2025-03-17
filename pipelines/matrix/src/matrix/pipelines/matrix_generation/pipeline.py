from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ARGO_GPU_NODE_MEDIUM, ArgoNode
from matrix.pipelines.modelling.utils import partial_fold

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix generation pipeline."""

    model_name = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")["model_name"]

    # Load cross-validation information
    cross_validation_settings = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation")
    n_cross_val_folds = cross_validation_settings.get("n_cross_val_folds")

    # Initial nodes computing matrix pairs and flags
    pipelines = []

    # Add shared nodes
    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.enrich_embeddings,
                    inputs=[
                        "embeddings.feat.nodes",
                        "integration.int.drug_list.nodes.norm@spark",
                        "integration.int.disease_list.nodes.norm@spark",
                    ],
                    outputs="matrix_generation.feat.nodes@spark",
                    name="enrich_matrix_embeddings",
                ),
            ]
        )
    )

    # Nodes generating scores for each fold and model
    for fold in range(n_cross_val_folds + 1):  # NOTE: final fold is full training data
        # For each fold, generate the pairs
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=partial_fold(nodes.generate_pairs, fold, arg_name="known_pairs"),
                        inputs={
                            "known_pairs": "modelling.model_input.splits",
                            "drugs": "integration.int.drug_list.nodes.norm@pandas",
                            "diseases": "integration.int.disease_list.nodes.norm@pandas",
                            "graph": "matrix_generation.feat.nodes@kg",
                            "clinical_trials": "integration.int.ec_clinical_trails.edges.norm@pandas",
                        },
                        outputs=f"matrix_generation.prm.fold_{fold}.matrix_pairs",
                        name=f"generate_matrix_pairs_fold_{fold}",
                    )
                ]
            )
        )

        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.make_predictions_and_sort,
                        inputs=[
                            "matrix_generation.feat.nodes@kg",
                            f"matrix_generation.prm.fold_{fold}.matrix_pairs",
                            f"modelling.fold_{fold}.model_input.transformers",
                            f"modelling.fold_{fold}.models.model",
                            f"params:modelling.{model_name}.model_options.model_tuning_args.features",
                            "params:matrix_generation.treat_score_col_name",
                            "params:matrix_generation.not_treat_score_col_name",
                            "params:matrix_generation.unknown_score_col_name",
                            "params:matrix_generation.matrix_generation_options.batch_by",
                        ],
                        outputs=f"matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                        name=f"make_predictions_and_sort_fold_{fold}",
                        argo_config=ARGO_GPU_NODE_MEDIUM,
                    ),
                    ArgoNode(
                        func=nodes.generate_report,
                        inputs=[
                            f"matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                            "params:matrix_generation.matrix_generation_options.n_reporting",
                            "integration.int.drug_list.nodes.norm@pandas",
                            "integration.int.disease_list.nodes.norm@pandas",
                            "params:matrix_generation.treat_score_col_name",
                            "params:matrix_generation.matrix",
                            "params:matrix_generation.run",
                        ],
                        outputs=f"matrix_generation.fold_{fold}.reporting.matrix_report",
                        name=f"generate_report_fold_{fold}",
                    ),
                ],
            )
        )

    return sum(pipelines)


def make_predictions_and_sort(
    graph: KnowledgeGraph,
    data: ps.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    treat_score_col_name: str,
    not_treat_score_col_name: str,
    unknown_score_col_name: str,
) -> ps.DataFrame:
    """Generate probability scores for drug-disease dataset.

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset with additional column containing the probability scores.
    """

    # Unpickle the model
    pkl = sc.binaryFiles(model)  # deserializing the model
    collection = pkl.collect()  # collecting the binary files into a list
    model = pickle.loads(collection[0][1])  # loading the collected model data
    model = sc.broadcast(model)  # broadcasting the model

    @F.pandas_udf(T.FloatType())
    def get_embedding_udf(column: pd.Series) -> pd.Series:
        return column.apply(lambda x: graph.get_embedding(x, default=pd.NA))

    # Collect embedding vectors
    data = data.withColumn("source_embedding", get_embedding_udf(F.col("source")))
    data = data.withColumn("target_embedding", get_embedding_udf(F.col("target")))

    # Retrieve rows with null embeddings
    # NOTE: This only happens in a rare scenario where the node synonymizer
    # provided an identifier for a node that does _not_ exist in our KG.
    # https://github.com/everycure-org/matrix/issues/409
    removed = data.filter(F.col("source_embedding").isNull() | F.col("target_embedding").isNull())

    if removed.count() > 0:
        logger.warning(f"Dropped {removed.count()} pairs during generation!")
        removed_pairs = (
            removed.select("source", "target").rdd.map(lambda r: f"({r['source']}, {r['target']})").collect()
        )
        logger.warning("Dropped: %s", ",".join(removed_pairs))

    # Drop rows without source/target embeddings
    data = data.dropna(subset=["source_embedding", "target_embedding"])

    # Return empty dataframe if all rows are dropped
    if data.count() == 0:
        return data.drop("source_embedding", "target_embedding")

    # Apply transformers to data (assuming this can work with PySpark)
    transformed = apply_transformers(data, transformers)

    # Extract features
    data_features = _extract_elements_in_list(transformed.columns, features, True)

    # Make predictions
    # https://medium.com/@smoothml/scikit-learn-predictions-on-apache-spark-3987feff44a2
    @F.pandas_udf(
        T.StructType(
            [
                T.StructField(not_treat_score_col_name, T.FloatType()),
                T.StructField(treat_score_col_name, T.FloatType()),
                T.StructField(unknown_score_col_name, T.FloatType()),
            ]
        )
    )
    def predict(*cols) -> pd.DataFrame:
        # Columns are passed as a tuple of Pandas Series'.
        # Combine into a Pandas DataFrame
        X = pd.concat(cols, axis=1)

        # Make predictions and select probabilities of positive class (1).
        predictions = model.value.predict_proba(X)
        # Return Pandas dataframe of predictions.
        return pd.DataFrame(
            predictions, columns=[not_treat_score_col_name, treat_score_col_name, unknown_score_col_name]
        )

    # Single prediction call
    preds = predict(*[F.col(f) for f in data_features])

    # Extract struct fields
    data = data.withColumns(
        {
            not_treat_score_col_name: preds[not_treat_score_col_name],
            treat_score_col_name: preds[treat_score_col_name],
            unknown_score_col_name: preds[unknown_score_col_name],
        }
    )

    data = data.drop("source_embedding", "target_embedding")

    data = data.orderBy(treat_score_col_name, ascending=False)
    data.withColumn("rank", range(1, data.count() + 1))
    data.withColumn("quantile_rank", data.select("rank") / data.count())

    return data


@inject_object()
def apply_transformers(
    data: ps.DataFrame,
    transformers: dict[str, dict[str, Union[_BaseImputer, list[str]]]],
) -> ps.DataFrame:
    """Function to apply fitted transformers to the data.

    Args:
        data: Data to transform.
        transformers: Dictionary of transformers.

    Returns:
        Transformed data.
    """
    data = data.with_columns(index=F.monotonically_increasing_id())
    for transformer in transformers.values():
        # Extract features for transformation
        features = transformer["features"]  # Assuming this is a list of column names

        features_selected = data.select(*features).toPandas()
        # The transformer is inherited from sklearn.preprocessing.FunctionTransformer
        transformed_values = transformer["transformer"].transform(features_selected)

        # Convert transformed values back to Spark DataFrame
        transformed_sdf = spark.createDataFrame(
            pd.DataFrame(
                transformed_values, columns=transformer["transformer"].get_feature_names_out(features_selected)
            )
        ).withColumn("index", F.monotonically_increasing_id())

        # Join transformed data back to original DataFrame
        data = (
            data.drop(*features)  # Drop old features
            .join(transformed_sdf, on="index", how="inner")
            .drop("index")  # Clean up index column
        )

    return data
