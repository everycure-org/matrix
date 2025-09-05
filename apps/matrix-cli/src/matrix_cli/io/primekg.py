"""
Python port of src/bin/primekg.rs.

Subcommands:
- build-edges: Transform a PrimeKG edges TSV into KGX-like edges with Biolink predicates.
- build-nodes: Join multiple node feature TSVs and map types/IDs to CURIEs.
- print-predicate-mappings: Infer mapping table between relation/display_relation and Biolink predicate; print as pretty JSON to stdout.

Examples:
  primekg build-edges -i kg.tsv -o edges.tsv
  primekg build-nodes -a drug_features.tsv -b disease_features.tsv -n nodes.tsv -o nodes_out.tsv
  primekg print-predicate-mappings -i primekg.tsv

"""

from __future__ import annotations

from pathlib import Path

import click
import polars as pl
from matrix_io.primekg import coalesce_duplicate_columns, mondo_grouped_exploded


def read_tsv_lazy(path: Path) -> pl.LazyFrame:
    """Read TSV to a LazyFrame."""
    return pl.scan_csv(
        path,
        separator=",",
        infer_schema_length=0,
        has_header=True,
        ignore_errors=True,
        truncate_ragged_lines=True,
    )


@click.group()
def main() -> None:
    """PrimeKG utilities (Python port)."""
    pass


@main.command("print-predicate-mappings")
@click.option("kg", "-i", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Input PrimeKG TSV")
def print_predicate_mappings_cmd(kg: Path) -> None:
    """Print mapping of relation/display_relation to Biolink predicate (JSON)."""
    orig_df = read_tsv_lazy(kg).with_columns(
        [
            pl.lit(None, dtype=pl.Utf8).alias("subject_aspect_qualifier"),
            pl.lit(None, dtype=pl.Utf8).alias("subject_direction_qualifier"),
            pl.lit(None, dtype=pl.Utf8).alias("predicate"),
            pl.lit(None, dtype=pl.Boolean).alias("negated"),
        ]
    )

    final_df = (
        pl.concat(
            [
                orig_df.filter(
                    pl.col("relation").str.contains(
                        "bioprocess_protein|cellcomp_protein|exposure_bioprocess|exposure_cellcomp|exposure_molfunc|exposure_protein|molfunc_protein|pathway_protein|protein_protein"
                    )
                ).with_columns([pl.lit("biolink:interacts_with").alias("predicate")]),
                orig_df.filter(pl.col("relation").str.contains("disease_protein|phenotype_protein")).with_columns(
                    [pl.lit("biolink:associated_with").alias("predicate")]
                ),
                orig_df.filter(
                    pl.col("relation").str.contains(
                        "anatomy_anatomy|bioprocess_bioprocess|cellcomp_cellcomp|disease_disease|exposure_exposure|molfunc_molfunc|pathway_pathway|phenotype_phenotype"
                    )
                ).with_columns([pl.lit("biolink:superclass_of").alias("predicate")]),
                orig_df.filter(pl.col("relation").str.contains("drug_effect")).with_columns(
                    [pl.lit("biolink:has_side_effect").alias("predicate")]
                ),
                orig_df.filter(pl.col("relation").str.contains("contraindication")).with_columns(
                    [pl.lit("biolink:contraindicated_in").alias("predicate")]
                ),
                orig_df.filter(pl.col("relation").str.contains("anatomy_protein_absent")).with_columns(
                    [pl.lit(True).alias("negated"), pl.lit("biolink:expressed_in").alias("predicate")]
                ),
                orig_df.filter(pl.col("relation").str.contains("anatomy_protein_present")).with_columns(
                    [pl.lit("biolink:expressed_in").alias("predicate")]
                ),
                orig_df.filter(pl.col("relation").str.contains("disease_phenotype_negative")).with_columns(
                    [pl.lit(True).alias("negated"), pl.lit("biolink:has_phenotype").alias("predicate")]
                ),
                orig_df.filter(pl.col("relation").str.contains("disease_phenotype_positive")).with_columns(
                    [pl.lit("biolink:has_phenotype").alias("predicate")]
                ),
                orig_df.filter(pl.col("relation").str.contains("exposure_disease")).with_columns(
                    [pl.lit("biolink:correlated_with").alias("predicate")]
                ),
                orig_df.filter(pl.col("relation").str.contains("indication")).with_columns([pl.lit("biolink:treats").alias("predicate")]),
                orig_df.filter(pl.col("relation").str.contains("off-label use")).with_columns(
                    [pl.lit("biolink:applied_to_treat").alias("predicate")]
                ),
                orig_df.filter(pl.col("relation").str.contains("drug_drug")).with_columns(
                    [pl.lit("biolink:directly_physically_interacts_with").alias("predicate")]
                ),
                orig_df.filter(
                    (pl.col("relation") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("enzyme"))
                ).with_columns(
                    [
                        pl.lit("amount").alias("subject_aspect_qualifier"),
                        pl.lit("biolink:affected_by").alias("predicate"),
                    ]
                ),
                orig_df.filter(
                    (pl.col("relation") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("target"))
                ).with_columns(
                    [
                        pl.lit("biolink:directly_physically_interacts_with").alias("predicate"),
                    ]
                ),
                orig_df.filter(
                    (pl.col("relation") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("carrier"))
                ).with_columns(
                    [
                        pl.lit("biolink:affected_by").alias("predicate"),
                    ]
                ),
                orig_df.filter(
                    (pl.col("relation") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("transporter"))
                ).with_columns(
                    [
                        pl.lit("transport").alias("subject_aspect_qualifier"),
                        pl.lit("increased").alias("subject_direction_qualifier"),
                        pl.lit("biolink:affected_by").alias("predicate"),
                    ]
                ),
            ]
        )
        .unique(subset=["relation", "display_relation", "predicate"], keep="first")
        .select(["relation", "display_relation", "predicate", "subject_aspect_qualifier", "subject_direction_qualifier", "negated"])
        .collect()
    )

    print(final_df.write_json())


@main.command("build-nodes")
@click.option("drug_features", "-a", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Drug features TSV")
@click.option(
    "disease_features", "-b", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Disease features TSV"
)
@click.option("nodes", "-n", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Nodes TSV")
@click.option(
    "output", "-o", type=click.Path(dir_okay=False, writable=True, path_type=Path), required=True, help="Output TSV (tab-separated)"
)
def build_nodes_cmd(drug_features: Path, disease_features: Path, nodes: Path, output: Path) -> None:
    """Build the nodes tsv file."""
    main = pl.DataFrame({"node_index": pl.Series([], dtype=pl.Utf8)})

    nodes_df = read_tsv_lazy(nodes)
    main = main.lazy().join(nodes_df, on=["node_index"], how="full", coalesce=True).collect()

    drug_features_df = read_tsv_lazy(drug_features)
    main = main.lazy().join(drug_features_df, on=["node_index"], how="full", coalesce=True).collect()

    main = coalesce_duplicate_columns(main, keep=["node_index"])  # preserves first non-null among suffixed columns

    disease_features_df = read_tsv_lazy(disease_features)
    main = main.lazy().join(disease_features_df, on=["node_index"], how="full", coalesce=True).collect()
    main = coalesce_duplicate_columns(main, keep=["node_index"])  # again after join

    # Type/category mapping and CURIE formatting
    main = (
        main.lazy()
        .with_columns(
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
                        [pl.col("node_source"), pl.col("node_id").cast(pl.Utf8).str.pad_start(7, "0")], separator=":", ignore_nulls=True
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
                        [pl.lit("MONDO"), pl.col("mondo_id").cast(pl.Utf8).str.pad_start(7, "0")], separator=":", ignore_nulls=True
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
        .collect()
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    main.write_csv(output, separator="\t")


@main.command("build-edges")
@click.option("kg", "-i", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Input PrimeKG TSV")
@click.option(
    "output", "-o", type=click.Path(dir_okay=False, writable=True, path_type=Path), required=True, help="Output edges TSV (tab-separated)"
)
def build_edges_cmd(kg: Path, output: Path) -> None:
    """Build the edges tsv file."""
    edges = read_tsv_lazy(kg).collect()

    edges = mondo_grouped_exploded(edges, "x")
    edges = mondo_grouped_exploded(edges, "y")

    edges = (
        edges.lazy()
        .with_columns(
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
        .collect()
    )

    # print(edges.shape)

    orig_df = (
        # subject CURIE formatting
        edges.lazy()
        .with_columns(
            [
                pl.when(pl.col("x_source") == pl.lit("NCBI"))
                .then(pl.concat_str([pl.col("x_source"), pl.col("x_id")], separator="Gene:", ignore_nulls=True))
                .otherwise(pl.col("x_source"))
                .alias("subject"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("x_source") == pl.lit("REACTOME"))
                .then(pl.concat_str([pl.lit("REACT"), pl.col("x_id")], separator=":", ignore_nulls=True))
                .otherwise(pl.col("subject"))
                .alias("subject"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("x_source").str.contains("HPO|MONDO|UBERON"))
                .then(
                    pl.concat_str(
                        [pl.col("x_source"), pl.col("x_id").cast(pl.Utf8).str.pad_start(7, "0")], separator=":", ignore_nulls=True
                    )
                )
                .otherwise(pl.col("subject"))
                .alias("subject"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("x_source").str.contains("CTD|GO|DrugBank"))
                .then(pl.concat_str([pl.col("x_source"), pl.col("x_id")], separator=":", ignore_nulls=True))
                .otherwise(pl.col("subject"))
                .alias("subject"),
            ]
        )
        # object CURIE formatting
        .with_columns(
            [
                pl.when(pl.col("y_source") == pl.lit("NCBI"))
                .then(pl.concat_str([pl.col("y_source"), pl.col("y_id")], separator="Gene:", ignore_nulls=True))
                .otherwise(pl.col("y_source"))
                .alias("object"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("y_source") == pl.lit("REACTOME"))
                .then(pl.concat_str([pl.lit("REACT"), pl.col("y_id")], separator=":", ignore_nulls=True))
                .otherwise(pl.col("object"))
                .alias("object"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("y_source").str.contains("HPO|MONDO|UBERON"))
                .then(
                    pl.concat_str(
                        [pl.col("y_source"), pl.col("y_id").cast(pl.Utf8).str.pad_start(7, "0")], separator=":", ignore_nulls=True
                    )
                )
                .otherwise(pl.col("object"))
                .alias("object"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("y_source").str.contains("CTD|GO|DrugBank"))
                .then(pl.concat_str([pl.col("y_source"), pl.col("y_id")], separator=":", ignore_nulls=True))
                .otherwise(pl.col("object"))
                .alias("object"),
            ]
        )
        .drop(["x_index", "x_id", "x_type", "x_name", "x_source", "y_index", "y_id", "y_type", "y_name", "y_source"], strict=False)
        .collect()
        .rename({"relation": "predicate"})
    )

    # print(orig_df.shape)

    final_lf = (
        pl.concat(
            [
                orig_df.filter(
                    pl.col("predicate").str.contains(
                        "bioprocess_protein|cellcomp_protein|exposure_bioprocess|exposure_cellcomp|exposure_molfunc|exposure_protein|molfunc_protein|pathway_protein"
                    )
                ).with_columns([pl.lit("biolink:interacts_with").alias("predicate")]),
                orig_df.filter(pl.col("predicate").str.contains("disease_protein|phenotype_protein")).with_columns(
                    [pl.lit("biolink:associated_with").alias("predicate")]
                ),
                orig_df.filter(
                    pl.col("predicate").str.contains(
                        "anatomy_anatomy|bioprocess_bioprocess|cellcomp_cellcomp|disease_disease|exposure_exposure|molfunc_molfunc|pathway_pathway|phenotype_phenotype"
                    )
                ).with_columns([pl.lit("biolink:superclass_of").alias("predicate")]),
                orig_df.filter(pl.col("predicate") == pl.lit("protein_protein")).with_columns(
                    [pl.lit("biolink:interacts_with").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("drug_effect")).with_columns(
                    [pl.lit("biolink:has_side_effect").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("contraindication")).with_columns(
                    [pl.lit("biolink:contraindicated_in").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("anatomy_protein_absent")).with_columns(
                    [pl.lit(True).alias("negated"), pl.lit("biolink:expressed_in").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("anatomy_protein_present")).with_columns(
                    [pl.lit("biolink:expressed_in").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("disease_phenotype_negative")).with_columns(
                    [pl.lit(True).alias("negated"), pl.lit("biolink:has_phenotype").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("disease_phenotype_positive")).with_columns(
                    [pl.lit("biolink:has_phenotype").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("exposure_disease")).with_columns(
                    [pl.lit("biolink:correlated_with").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("indication")).with_columns([pl.lit("biolink:treats").alias("predicate")]),
                orig_df.filter(pl.col("predicate") == pl.lit("off-label use")).with_columns(
                    [pl.lit("biolink:applied_to_treat").alias("predicate")]
                ),
                orig_df.filter(pl.col("predicate") == pl.lit("drug_drug")).with_columns(
                    [pl.lit("biolink:directly_physically_interacts_with").alias("predicate")]
                ),
                orig_df.filter(
                    (pl.col("predicate") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("enzyme"))
                ).with_columns(
                    [pl.lit("biolink:amount").alias("subject_aspect_qualifier"), pl.lit("biolink:affected_by").alias("predicate")]
                ),
                orig_df.filter(
                    (pl.col("predicate") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("target"))
                ).with_columns([pl.lit("biolink:directly_physically_interacts_with").alias("predicate")]),
                orig_df.filter(
                    (pl.col("predicate") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("carrier"))
                ).with_columns([pl.lit("biolink:affected_by").alias("predicate")]),
                orig_df.filter(
                    (pl.col("predicate") == pl.lit("drug_protein")) & (pl.col("display_relation") == pl.lit("transporter"))
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

    # print(final_lf.shape)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_lf.write_csv(output, separator="\t")


if __name__ == "__main__":
    main()
