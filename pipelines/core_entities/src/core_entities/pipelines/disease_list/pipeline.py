from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_ingestion_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.ingest_disease_list,
                inputs="disease_mondo.prm.disease_list",
                outputs="primary.disease_list",
                name="ingest_disease_list",
            ),
            node(
                func=nodes.ingest_curated_disease_list,
                inputs="raw.curated_disease_list",
                outputs="primary.curated_disease_list",
                name="ingest_curated_disease_list",
            ),
            node(
                func=nodes.ingest_disease_categories,
                inputs="raw.disease_categories",
                outputs="primary.disease_categories",
                name="ingest_disease_categories",
            ),
            node(
                func=nodes.ingest_disease_umn,
                inputs="raw.disease_umn",
                outputs="primary.disease_umn",
                name="ingest_disease_umn",
            ),
            node(
                func=nodes.ingest_disease_prevalence,
                inputs="raw.disease_prevalence",
                outputs="primary.disease_prevalence",
                name="ingest_disease_prevalence",
            ),
            node(
                func=nodes.ingest_disease_txgnn,
                inputs="raw.disease_txgnn",
                outputs="primary.disease_txgnn",
                name="ingest_disease_txgnn",
            ),
            node(
                func=nodes.ingest_disease_obsolete,
                inputs="disease_mondo.prm.mondo_obsoletes",
                outputs="primary.disease_obsolete",
                name="ingest_disease_obsolete",
            ),
            node(
                func=nodes.ingest_orchard_reviews,
                inputs="raw.orchard_reviews",
                outputs="primary.orchard_reviews",
                name="ingest_orchard_reviews",
            ),
            node(
                func=nodes.ingest_disease_name_patch,
                inputs="input.disease_name_patch",
                outputs="primary.disease_name_patch",
                name="ingest_disease_name_patch",
            ),
            node(
                func=nodes.ingest_manual_disease_remapping,
                inputs="input.manual_disease_remapping",
                outputs="primary.manual_disease_remapping",
                name="ingest_manual_disease_remapping",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            create_ingestion_pipeline(),
            # ----------
            # Merging zone
            # ----------
            node(
                func=nodes.merge_disease_lists,
                inputs={
                    "disease_list": "primary.disease_list",
                    "disease_categories": "primary.disease_categories",
                    "disease_umn": "primary.disease_umn",
                    "disease_prevalence": "primary.disease_prevalence",
                    "disease_txgnn": "primary.disease_txgnn",
                    "curated_disease_list": "primary.curated_disease_list",
                },
                outputs="primary.disease_list_merged",
                name="merge_disease_lists",
            ),
            # ----------
            # Patch names
            # ----------
            node(
                func=nodes.apply_name_patch,
                inputs={
                    "disease_list": "primary.disease_list_merged",
                    "disease_name_patch": "primary.disease_name_patch",
                },
                outputs="primary.disease_list_patched",
                name="disease_patch_name",
            ),
            # ----------
            # Release
            # ----------
            node(
                func=nodes.format_disease_list,
                inputs={
                    "disease_list": "primary.disease_list_patched",
                    "release_columns": "params:disease_list.release_columns",
                },
                outputs="primary.disease_list_formatted",
                name="format_disease_list",
            ),
            node(
                func=nodes.migrate_diseases_with_dangling_reviews,
                inputs={
                    "disease_list": "primary.disease_list_formatted",
                    "disease_obsolete": "primary.disease_obsolete",
                    "orchard_reviews": "primary.orchard_reviews",
                    "manual_disease_remapping": "primary.manual_disease_remapping",
                },
                outputs=dict(
                    disease_list_with_migrations_parquet="primary.release.disease_list_parquet",
                    disease_list_with_migrations_tsv="primary.release.disease_list_tsv",
                    reviews_to_map="primary.release.reviews_to_map",
                ),
                name="migrate_diseases_with_dangling_reviews",
            ),
        ]
    )


def create_publish_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.publish_disease_list,
                inputs="primary.release.disease_list_parquet",
                outputs=dict(
                    disease_list_parquet="primary.published.disease_list",
                    disease_list_tsv="primary.published.disease_list_tsv",
                    disease_list_bq="primary.published.disease_list_bq",
                    disease_list_bq_latest="primary.published.disease_list_bq_latest",
                ),
                name="publish_disease_list",
            ),
        ]
    )


def create_publish_hf_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.merge_mondo_and_ec_disease_list,
                inputs={
                    "mondo_disease_list": "disease_mondo.prm.disease_list",
                    "ec_disease_list": "primary.release.disease_list_parquet",
                },
                outputs="primary.published.disease_list_hf",
                name="publish_disease_list_hf",
            ),
        ]
    )
