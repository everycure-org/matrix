from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # ----------
            # Ingestion
            # ----------
            node(
                func=nodes.ingest_curated_drug_list,
                inputs="raw.curated_drug_list",
                outputs="primary.curated_drug_list",
                name="ingest_curated_drug_list",
            ),
            node(
                func=nodes.ingest_drugbank_drug_list,
                inputs="raw.drugbank_drug_list",
                outputs="primary.drugbank_drug_list",
                name="ingest_drugbank_drug_list",
            ),
            node(
                func=nodes.ingest_drugbank_salt_list,
                inputs="raw.drugbank_salt_list",
                outputs="primary.drugbank_salt_list",
                name="ingest_drugbank_salt_list",
            ),
            node(
                func=nodes.ingest_drugbank_drug_atc,
                inputs="raw.drugbank_drug_atc",
                outputs="primary.drugbank_drug_atc",
                name="ingest_drugbank_drug_atc",
            ),
            node(
                func=nodes.ingest_drugbank_salt_atc,
                inputs="raw.drugbank_salt_atc",
                outputs="primary.drugbank_salt_atc",
                name="ingest_drugbank_salt_atc",
            ),
            node(
                func=nodes.ingest_drugbank_pure_atc,
                inputs="raw.drugbank_pure_atc",
                outputs="primary.drugbank_pure_atc",
                name="ingest_drugbank_pure_atc",
            ),
            # ----------
            # Drug resolution
            # ----------
            node(
                func=nodes.resolve_drug_curies,
                inputs={
                    "curated_drug_list": "primary.curated_drug_list",
                    "name_resolver_base_url": "params:drug_list.name_resolver.base_url",
                    "name_resolver_path": "params:drug_list.name_resolver.path",
                    "accepted_drug_categories": "params:drug_list.name_resolver.accepted_drug_categories",
                    "parallelism": "params:drug_list.name_resolver.parallelism",
                },
                outputs=dict(
                    name_resolved_curies="primary.drug_curies",
                    name_resolver_version="primary.name_resolver_version",
                ),
                name="resolve_drug_curies",
            ),
            node(
                func=nodes.normalize_drug_curies,
                inputs={
                    "drug_curies": "primary.drug_curies",
                    "node_normalizer_base_url": "params:drug_list.node_normalizer.base_url",
                    "node_normalizer_path": "params:drug_list.node_normalizer.path",
                },
                outputs=dict(
                    normalized_drug_curies="primary.normalized_drug_curies",
                    nodenorm_version="primary.nodenorm_version",
                ),
                name="normalize_drug_curies",
            ),
            node(
                func=nodes.union_drugbank_lists,
                inputs={
                    "drugbank_drug_list": "primary.drugbank_drug_list",
                    "drugbank_salt_list": "primary.drugbank_salt_list",
                },
                outputs="primary.drugbank_union_list",
                name="union_drugbank_lists",
            ),
            node(
                func=nodes.resolve_drugbank_ids,
                inputs={
                    "curated_drug_list": "primary.curated_drug_list",
                    "drugbank_union_list": "primary.drugbank_union_list",
                },
                outputs="primary.drug_list_with_drugbank_id",
                name="resolve_drugbank_ids",
            ),
            node(
                func=nodes.resolve_atc_codes,
                inputs={
                    "drug_list_with_drugbank_id": "primary.drug_list_with_drugbank_id",
                    "drugbank_drug_atc": "primary.drugbank_drug_atc",
                    "drugbank_salt_atc": "primary.drugbank_salt_atc",
                    "drugbank_pure_atc": "primary.drugbank_pure_atc",
                },
                outputs="primary.drug_list_with_atc_codes",
                name="resolve_atc_codes",
            ),
            node(
                func=nodes.merge_drug_lists,
                inputs={
                    "curated_drug_list": "primary.curated_drug_list",
                    "normalized_drug_curies": "primary.normalized_drug_curies",
                    "drug_list_with_atc_codes": "primary.drug_list_with_atc_codes",
                    "release_columns": "params:drug_list.release_columns",
                    "drug_exception_list": "params:drug_list.drug_exception_list",
                },
                outputs=[
                    "primary.release.drug_list_parquet",
                    "primary.release.drug_list_tsv",
                ],
                name="merge_drug_lists",
            ),
        ]
    )


def create_publish_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.publish_drug_list,
                inputs="primary.release.drug_list_parquet",
                outputs=dict(
                    drug_list_parquet="primary.published.drug_list",
                    drug_list_tsv="primary.published.drug_list_tsv",
                    drug_list_bq="primary.published.drug_list_bq",
                    drug_list_bq_latest="primary.published.drug_list_bq_latest",
                ),
                name="publish_drug_list",
            ),
        ]
    )
