import polars as pl
from matrix_io_utils.primekg import coalesce_duplicate_columns, fix_curies, mondo_grouped_exploded


def build_nodes(nodes: pl.DataFrame, drug_features: pl.DataFrame, disease_features: pl.DataFrame) -> pl.DataFrame:
    main = pl.DataFrame({"node_index": pl.Series([], dtype=pl.Utf8)})

    main = main.join(nodes, on=["node_index"], how="full", coalesce=True)

    main = main.join(drug_features, on=["node_index"], how="full", coalesce=True)
    main = coalesce_duplicate_columns(main, keep=["node_index"])  # preserves first non-null among suffixed columns

    main = main.join(disease_features, on=["node_index"], how="full", coalesce=True)
    main = coalesce_duplicate_columns(main, keep=["node_index"])  # again after join

    # Type/category mapping and CURIE formatting
    main = (
        main.with_columns(
            [
                pl.when(pl.col("node_source").str.contains("NCBI"))
                .then(pl.concat_str([pl.col("node_source"), pl.col("node_id")], separator="Gene:", ignore_nulls=True))
                .otherwise(pl.col("node_source"))
                .alias("node_source"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_source") == pl.lit("REACTOME"))
                .then(pl.concat_str([pl.lit("REACT"), pl.col("node_id")], separator=":", ignore_nulls=True))
                .otherwise(pl.col("node_source"))
                .alias("node_source"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_source").str.contains("HPO|MONDO|UBERON"))
                .then(
                    pl.concat_str(
                        [pl.col("node_source"), pl.col("node_id").cast(pl.Utf8).str.pad_start(7, "0")],
                        separator=":",
                        ignore_nulls=True,
                    )
                )
                .otherwise(pl.col("node_source"))
                .alias("node_source"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_source").str.contains("CTD|GO|DrugBank"))
                .then(pl.concat_str([pl.col("node_source"), pl.col("node_id")], separator=":", ignore_nulls=True))
                .otherwise(pl.col("node_source"))
                .alias("node_source"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_source").str.contains("MONDO_grouped", literal=True))
                .then(
                    pl.concat_str(
                        [pl.lit("MONDO"), pl.col("mondo_id").cast(pl.Utf8).str.pad_start(7, "0")],
                        separator=":",
                        ignore_nulls=True,
                    )
                )
                .otherwise(pl.col("node_source"))
                .alias("node_source"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("exposure"))
                .then(pl.lit("biolink:ChemicalExposure"))
                .otherwise(pl.col("node_type"))
                .alias("node_type")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("effect/phenotype"))
                .then(pl.lit("biolink:PhenotypicFeature"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("molecular_function"))
                .then(pl.lit("biolink:MolecularActivity"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("cellular_component"))
                .then(pl.lit("biolink:CellularComponent"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("biological_process"))
                .then(pl.lit("biolink:BiologicalProcess"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("pathway"))
                .then(pl.lit("biolink:Pathway"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("gene/protein"))
                .then(pl.lit("biolink:Gene"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("disease"))
                .then(pl.lit("biolink:Disease"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("drug"))
                .then(pl.lit("biolink:SmallMolecule"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("node_type") == pl.lit("anatomy"))
                .then(pl.lit("biolink:GrossAnatomicalStructure"))
                .otherwise(pl.col("node_type"))
                .alias("node_type"),
            ]
        )
        .drop(["node_id", "node_index"], strict=False)
        .rename({"node_source": "id", "node_name": "name", "category": "drug_category"})
        .rename({"node_type": "category"})
    )
    return main


def build_edges(edges: pl.DataFrame) -> pl.DataFrame:
    """Build the edges tsv file."""

    edges = edges.lazy().with_columns(
        [
            pl.lit("knowledge_assertion").alias("knowledge_level"),
            pl.lit("not_provided").alias("agent_type"),
            pl.lit("infores:primekg").alias("primary_knowledge_source"),
            pl.lit(None, dtype=pl.Utf8).alias("aggregator_knowledge_source"),
            pl.lit(None, dtype=pl.Utf8).alias("original_subject"),
            pl.lit(None, dtype=pl.Utf8).alias("original_object"),
            pl.lit(None, dtype=pl.Boolean).alias("negated"),
            pl.lit(None, dtype=pl.Utf8).alias("publications"),
            pl.lit(None, dtype=pl.Utf8).alias("subject_aspect_qualifier"),
            pl.lit(None, dtype=pl.Utf8).alias("subject_direction_qualifier"),
            pl.lit(None, dtype=pl.Utf8).alias("object_aspect_qualifier"),
            pl.lit(None, dtype=pl.Utf8).alias("object_direction_qualifier"),
            pl.lit(None, dtype=pl.Utf8).alias("upstream_data_source"),
        ]
    )

    edges = mondo_grouped_exploded(edges)
    edges = fix_curies(edges)

    edges = (
        edges.drop(
            ["x_index", "x_id", "x_type", "x_name", "x_source", "y_index", "y_id", "y_type", "y_name", "y_source"],
            strict=False,
        ).rename({"relation": "predicate"})
    ).collect()

    final_df = (
        pl.concat(
            [
                edges.filter(
                    pl.col("predicate").str.contains(
                        "bioprocess_protein|cellcomp_protein|exposure_bioprocess|exposure_cellcomp|exposure_molfunc|exposure_protein|molfunc_protein|pathway_protein"
                    )
                ).with_columns([pl.lit("biolink:interacts_with").alias("predicate")]),
                edges.filter(pl.col("predicate").str.contains("disease_protein|phenotype_protein")).with_columns(
                    [pl.lit("biolink:associated_with").alias("predicate")]
                ),
                edges.filter(
                    pl.col("predicate").str.contains(
                        "anatomy_anatomy|bioprocess_bioprocess|cellcomp_cellcomp|disease_disease|exposure_exposure|molfunc_molfunc|pathway_pathway|phenotype_phenotype"
                    )
                ).with_columns([pl.lit("biolink:superclass_of").alias("predicate")]),
                edges.filter(pl.col("predicate") == pl.lit("protein_protein")).with_columns(
                    [pl.lit("biolink:interacts_with").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("drug_effect")).with_columns(
                    [pl.lit("biolink:has_side_effect").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("contraindication")).with_columns(
                    [pl.lit("biolink:contraindicated_in").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("anatomy_protein_absent")).with_columns(
                    [pl.lit(True).alias("negated"), pl.lit("biolink:expressed_in").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("anatomy_protein_present")).with_columns(
                    [pl.lit("biolink:expressed_in").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("disease_phenotype_negative")).with_columns(
                    [pl.lit(True).alias("negated"), pl.lit("biolink:has_phenotype").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("disease_phenotype_positive")).with_columns(
                    [pl.lit("biolink:has_phenotype").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("exposure_disease")).with_columns(
                    [pl.lit("biolink:correlated_with").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("indication")).with_columns(
                    [pl.lit("biolink:treats").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("off-label use")).with_columns(
                    [pl.lit("biolink:applied_to_treat").alias("predicate")]
                ),
                edges.filter(pl.col("predicate") == pl.lit("drug_drug")).with_columns(
                    [pl.lit("biolink:directly_physically_interacts_with").alias("predicate")]
                ),
                edges.filter(
                    (pl.col("predicate") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("enzyme"))
                ).with_columns(
                    [
                        pl.lit("biolink:amount").alias("subject_aspect_qualifier"),
                        pl.lit("biolink:affected_by").alias("predicate"),
                    ]
                ),
                edges.filter(
                    (pl.col("predicate") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("target"))
                ).with_columns([pl.lit("biolink:directly_physically_interacts_with").alias("predicate")]),
                edges.filter(
                    (pl.col("predicate") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("carrier"))
                ).with_columns([pl.lit("biolink:affected_by").alias("predicate")]),
                edges.filter(
                    (pl.col("predicate") == pl.lit("drug_protein"))
                    & (pl.col("display_relation") == pl.lit("transporter"))
                ).with_columns(
                    [
                        pl.lit("transport").alias("subject_aspect_qualifier"),
                        pl.lit("increased").alias("subject_direction_qualifier"),
                        pl.lit("biolink:affected_by").alias("predicate"),
                    ]
                ),
            ]
        )
        .drop(["display_relation"], strict=False)
        .unique(subset=["subject", "predicate", "object"], keep="first")
    )

    return final_df
