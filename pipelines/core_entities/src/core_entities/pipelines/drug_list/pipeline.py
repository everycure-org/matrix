from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_ingestion_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.ingest_curated_drug_list,
                inputs="raw.curated_drug_list",
                outputs="primary.curated_drug_list",
                name="ingest_curated_drug_list",
            ),
            node(
                func=nodes.ingest_drugbank_identifiers,
                inputs="raw.drugbank_identifiers",
                outputs="primary.drugbank_identifiers",
                name="ingest_drugbank_identifiers",
            ),
            node(
                func=nodes.ingest_atc_labels,
                inputs="raw.atc_labels",
                outputs="primary.atc_labels",
                name="ingest_atc_labels",
            ),
            node(
                func=nodes.ingest_fda_drug_list,
                inputs="raw.fda_drug_list",
                outputs="primary.fda_drug_list",
                name="ingest_fda_drug_list",
            ),
        ]
    )


def create_resolution_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.resolve_fda_drugs_matches_to_drug_list_unfiltered,
                inputs={
                    "curated_drug_list": "primary.curated_drug_list",
                    "fda_drug_list": "primary.fda_drug_list",
                    "curated_drug_list_columns_to_use_for_matching": "params:drug_list.fda.curated_drug_list_columns_to_use_for_matching",
                    "fda_drug_list_columns_to_use_for_matching": "params:drug_list.fda.fda_drug_list_columns_to_use_for_matching",
                    "filter_curated_drug_list_params": "params:drug_list.fda.filter_curated_drug_list_params",
                },
                outputs="primary.fda_drug_labels_unfiltered",
                name="resolve_fda_drugs_matches_to_drug_list_unfiltered",
            ),
            node(
                func=nodes.resolve_fda_drugs_matches_to_drug_list_filtered,
                inputs={
                    "fda_drug_labels_unfiltered": "primary.fda_drug_labels_unfiltered",
                },
                outputs=[
                    "primary.fda_drug_labels_filtered_parquet",
                    "primary.fda_drug_labels_filtered_tsv",
                ],
                name="resolve_fda_drugs_matches_to_drug_list_filtered",
            ),
            node(
                func=nodes.resolve_fda_drugs_that_are_biosimilar_and_are_generic,
                inputs={
                    "fda_drug_labels_filtered": "primary.fda_drug_labels_filtered_parquet",
                    "fda_purple_book_params": "params:drug_list.fda_purple_book",
                    "fda_purple_book_data": "raw.fda_purple_book_data",
                },
                outputs=[
                    "primary.fda_drugs_filtered_biosimilar_parquet",
                    "primary.fda_drugs_filtered_biosimilar_tsv",
                ],
                name="resolve_fda_drugs_that_are_biosimilar_and_are_generic",
            ),
            node(
                func=nodes.resolve_fda_drugs_that_are_otc_monograph,
                inputs={
                    "fda_drug_labels_filtered": "primary.fda_drugs_filtered_biosimilar_parquet",
                    "fda_labels_params": "params:drug_list.fda_labels",
                },
                outputs=[
                    "primary.fda_drug_labels_filtered_including_otc_monograph_parquet",
                    "primary.fda_drug_labels_filtered_including_otc_monograph_tsv",
                    "primary.fda_generic_drug_list",
                ],
                name="resolve_fda_drugs_that_are_otc_monograph",
            ),
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
                func=nodes.resolve_drugbank_ids,
                inputs={
                    "curated_drug_list": "primary.curated_drug_list",
                    "drugbank_identifiers": "primary.drugbank_identifiers",
                },
                outputs="primary.drug_list_with_drugbank_id",
                name="resolve_drugbank_ids",
            ),
            node(
                func=nodes.resolve_atc_codes,
                inputs={"curated_drug_list": "primary.curated_drug_list", "atc_labels": "primary.atc_labels"},
                outputs="primary.drug_list_with_atc_codes",
                name="resolve_atc_codes",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            create_ingestion_pipeline(),
            create_resolution_pipeline(),
            node(
                func=nodes.merge_drug_lists,
                inputs={
                    "curated_drug_list": "primary.curated_drug_list",
                    "normalized_drug_curies": "primary.normalized_drug_curies",
                    "drug_list_with_atc_codes": "primary.drug_list_with_atc_codes",
                    "drug_list_with_drugbank_id": "primary.drug_list_with_drugbank_id",
                    "release_columns": "params:drug_list.release_columns",
                    "drug_exception_list": "params:drug_list.drug_exception_list",
                    "drug_list_with_fda_generic_drug_info": "primary.fda_generic_drug_list",
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


def create_publish_hf_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs="primary.release.drug_list_parquet",
                outputs="primary.published.drug_list_hf",
                name="publish_drug_list_hf",
            ),
        ]
    )
