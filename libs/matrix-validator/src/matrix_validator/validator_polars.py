"""Polars-based validator implementation."""

import json
import logging
from typing import Optional

import patito as pt
import polars as pl
from patito.exceptions import _display_error_loc

from matrix_validator import util
from matrix_validator.checks import (
    CURIE_REGEX,
    DELIMITED_BY_PIPES,
    NO_LEADING_WHITESPACE,
    NO_TRAILING_WHITESPACE,
    STARTS_WITH_BIOLINK_REGEX,
)
from matrix_validator.checks.check_column_contains_biolink_model_agent_type import (
    validate as check_column_contains_biolink_model_agent_type,
)
from matrix_validator.checks.check_column_contains_biolink_model_knowledge_level import (
    validate as check_column_contains_biolink_model_knowledge_level,
)
from matrix_validator.checks.check_column_contains_biolink_model_prefix import validate as check_column_contains_biolink_model_prefix
from matrix_validator.checks.check_column_enum import validate as check_column_enum
from matrix_validator.checks.check_column_is_delimited_by_pipes import validate as check_column_is_delimited_by_pipes
from matrix_validator.checks.check_column_is_valid_curie import validate as check_column_is_valid_curie
from matrix_validator.checks.check_column_no_leading_whitespace import validate as check_column_no_leading_whitespace
from matrix_validator.checks.check_column_no_trailing_whitespace import validate as check_column_no_trailing_whitespace
from matrix_validator.checks.check_column_range import validate as check_column_range
from matrix_validator.checks.check_column_starts_with_biolink import validate as check_column_starts_with_biolink
from matrix_validator.checks.check_edge_ids_in_node_ids import validate as check_edge_ids_in_node_ids
from matrix_validator.checks.check_node_id_and_category_with_biolink_preferred_prefixes import (
    validate as check_node_id_and_category_with_biolink_preferred_prefixes,
)
from matrix_validator.validator import Validator

from . import serialize_sets

logger = logging.getLogger("matrix-validator.polars")

BIOLINK_KNOWLEDGE_LEVEL_KEYS = util.get_biolink_model_knowledge_level_keys()
BIOLINK_AGENT_TYPE_KEYS = util.get_biolink_model_agent_type_keys()


def create_edges_schema(prefixes: list[str]):
    """Create a EdgeSchema with a dynamic list of allowed prefixes."""

    class EdgeSchema(pt.Model):
        """EdgeSchema derived from Patito."""

        # subject: str = pt.Field(constraints=[pt.field.str.contains(CURIE_REGEX), pt.field.str.contains_any(prefixes)])
        # predicate: str = pt.Field(constraints=[pt.field.str.contains(STARTS_WITH_BIOLINK_REGEX)])
        # object: str = pt.Field(constraints=[pt.field.str.contains(CURIE_REGEX), pt.field.str.contains_any(prefixes)])
        # knowledge_level: str = pt.Field(constraints=[pt.field.str.contains_any(BIOLINK_KNOWLEDGE_LEVEL_KEYS)])
        # agent_type: str = pt.Field(constraints=[pt.field.str.contains_any(BIOLINK_AGENT_TYPE_KEYS)])
        subject: str
        predicate: str
        object: str
        knowledge_level: str
        agent_type: str
        primary_knowledge_source: str
        aggregator_knowledge_source: str
        original_subject: str
        original_object: str
        publications: Optional[str]
        subject_aspect_qualifier: Optional[str]
        subject_direction_qualifier: Optional[str]
        object_aspect_qualifier: Optional[str]
        object_direction_qualifier: Optional[str]
        upstream_data_source: Optional[str]

    return EdgeSchema


def create_nodes_schema(prefixes: list[str]):
    """Create a NodeSchema with a dynamic list of allowed prefixes."""

    class NodeSchema(pt.Model):
        """NodeSchema derived from Patito."""

        # id: str = pt.Field(constraints=[pt.field.str.contains(CURIE_REGEX), pt.field.str.contains_any(prefixes)])
        # category: str = pt.Field(
        #     constraints=[
        #         pt.field.str.contains(STARTS_WITH_BIOLINK_REGEX),
        #         pt.field.str.contains(DELIMITED_BY_PIPES),
        #         pt.field.str.contains(NO_LEADING_WHITESPACE),
        #         pt.field.str.contains(NO_TRAILING_WHITESPACE),
        #     ]
        # )
        id: str
        category: str
        name: Optional[str]
        description: Optional[str]
        equivalent_identifiers: Optional[str]
        all_categories: Optional[str]
        publications: Optional[str]
        labels: Optional[str]
        international_resource_identifier: Optional[str]

    return NodeSchema


class ValidatorPolarsDataFrameImpl(Validator):
    """Polars-based validator implementation using Dataframes."""

    def __init__(self, nodes: pl.DataFrame, edges: pl.DataFrame, config=None):
        """Create a new instance of the polars-based validator."""
        super().__init__(nodes, edges, config)
        self._violations = []

    def validate(self, limit: int | None = None) -> int:
        """Validate inputs."""
        # do an initial schema check on nodes
        nodes_schema = create_nodes_schema(self.prefixes)
        nodes_violations = check_schema("nodes", nodes_schema, self._nodes.limit(10))
        self._violations.extend(nodes_violations)
        self._violations.extend(check_for_superfluous_columns("nodes", nodes_schema, self._nodes.limit(10)))

        # do an initial schema check on edges
        edges_schema = create_edges_schema(self.prefixes)
        edges_violations = check_schema("edges", edges_schema, self._edges.limit(10))
        self._violations.extend(edges_violations)
        self._violations.extend(check_for_superfluous_columns("edges", edges_schema, self._edges.limit(10)))

        # control flow here is if the above schema checks have violations (missing columns),
        # then the below checks will also fail due to run time errors instead of validation errors

        if not nodes_violations:
            self._violations.extend(self.validate_kg_nodes())

        if not edges_violations:
            self._violations.extend(self.validate_kg_edges())

        if not nodes_violations and not edges_violations:
            self._violations.extend(self.validate_nodes_and_edges())

        if len(self._violations) > 0:
            return 1
        return 0

    @property
    def violations(self) -> list:
        """Get the violations."""
        return self._violations

    def validate_kg_nodes(self, nodes_schema: pt.Model, limit: int | None = None):
        """Validate a knowledge graph using optional nodes TSV files."""
        logger.info("Validating nodes TSV...")

        validation_reports = []

        if limit:
            self._nodes = self._nodes.limit(limit)

        validation_reports.extend(run_config_range_checks(self._nodes, self.config_contents))
        validation_reports.extend(run_node_checks(self._nodes, self.prefixes, self.class_prefix_map))

        return validation_reports

    def validate_kg_edges(self, limit: int | None = None):
        """Validate a knowledge graph using optional edges TSV files."""
        logger.info("Validating edges TSV...")

        validation_reports = []

        if limit:
            self.edges = self._edges.limit(limit)

        validation_reports.extend(run_config_range_checks(self._edges, self.config_contents))
        validation_reports.extend(run_edge_checks(self._edges, self.prefixes))

        return validation_reports

    def validate_nodes_and_edges(self, limit: int | None = None):
        """Validate a knowledge graph nodes vs edges."""
        logger.info("Validating nodes & edges")
        validation_reports = []

        edges_df = self._edges.select([pl.col("subject"), pl.col("predicate"), pl.col("object")])

        unique_edge_ids = (
            pl.concat(
                items=[edges_df.select(pl.col("subject").alias("id")), edges_df.select(pl.col("object").alias("id"))],
                how="vertical",
                parallel=True,
            )
            .unique()
            .get_column("id")
            .to_list()
        )

        logger.info("collecting counts")

        # If limit is specified, apply it to both nodes and edges
        if limit:
            self._nodes = self._nodes.limit(limit)

        validation_reports.extend(analyze_edge_types(self._nodes, edges_df, unique_edge_ids))

        return validation_reports


class ValidatorPolarsFileImpl(Validator):
    """Polars-based validator implementation using files."""

    def __init__(self, nodes_file_path, edges_file_path, config=None):
        """Create a new instance of the polars-based validator."""
        super().__init__(nodes_file_path, edges_file_path, config)
        # Set a default report directory if none is provided

    def validate(self, limit: int | None = None) -> int:
        """Validate a knowledge graph as nodes and edges KGX TSV files."""
        violations = []

        # do an initial schema check on nodes
        nodes_schema = create_nodes_schema(self.prefixes)
        nodes_df = pl.scan_csv(self._nodes, separator="\t", has_header=True, ignore_errors=True, low_memory=True).limit(10).collect()
        nodes_violations = check_schema("nodes", nodes_schema, nodes_df)
        violations.extend(nodes_violations)
        violations.extend(check_for_superfluous_columns("nodes", nodes_schema, nodes_df))

        # do an initial schema check on edges
        edges_schema = create_edges_schema(self.prefixes)
        edges_df = pl.scan_csv(self._nodes, separator="\t", has_header=True, ignore_errors=True, low_memory=True).limit(10).collect()
        edges_violations = check_schema("edges", edges_schema, edges_df)
        violations.extend(edges_violations)
        violations.extend(check_for_superfluous_columns("edges", edges_schema, edges_df))

        # control flow here is if the above schema checks have violations (missing columns),
        # then the below checks will also fail due to run time errors instead of validation errors

        if not nodes_violations:
            violations.extend(self.validate_kg_nodes(limit))

        if not edges_violations:
            violations.extend(self.validate_kg_edges(limit))

        if not nodes_violations and not edges_violations:
            violations.extend(self.validate_nodes_and_edges(limit))

        # Write validation report
        self.write_output(violations)

        if len(violations) > 0:
            return 1
        return 0

    def validate_kg_nodes(self, limit):
        """Validate a knowledge graph using optional nodes TSV files."""
        logger.info("Validating nodes TSV...")

        validation_reports = []

        main_df = pl.scan_csv(self._nodes, separator="\t", has_header=True, ignore_errors=True, low_memory=True)

        if limit:
            df = main_df.limit(limit).collect()
        else:
            df = main_df.collect()

        validation_reports.extend(run_config_range_checks(df, self.config_contents))
        validation_reports.extend(run_node_checks(df, self.prefixes, self.class_prefix_map))

        return validation_reports

    def validate_kg_edges(self, limit):
        """Validate a knowledge graph using optional edges TSV files."""
        logger.info("Validating edges TSV...")

        validation_reports = []

        main_df = pl.scan_csv(self._edges, separator="\t", has_header=True, ignore_errors=True, low_memory=True)

        if limit:
            df = main_df.limit(limit).collect()
        else:
            df = main_df.collect()

        validation_reports.extend(run_config_range_checks(df, self.config_contents))
        validation_reports.extend(run_edge_checks(df, self.prefixes))

        return validation_reports

    def validate_nodes_and_edges(self, limit):
        """Validate a knowledge graph nodes vs edges."""
        logger.info("Validating nodes & edges")

        validation_reports = []

        edges_df = (
            pl.scan_csv(self._edges, separator="\t", has_header=True, ignore_errors=False, low_memory=True)
            .select([pl.col("subject"), pl.col("predicate"), pl.col("object")])
            .collect()
        )
        unique_edge_ids = (
            pl.concat(
                items=[edges_df.select(pl.col("subject").alias("id")), edges_df.select(pl.col("object").alias("id"))],
                how="vertical",
                parallel=True,
            )
            .unique()
            .get_column("id")
            .to_list()
        )

        logger.info("collecting counts")

        # Load node data
        nodes_df = pl.scan_csv(self._nodes, separator="\t", has_header=True, ignore_errors=False, low_memory=True)

        # If limit is specified, apply it to both nodes and edges
        if limit:
            nodes_df = nodes_df.limit(limit)

        # Collect node data
        nodes_df = nodes_df.collect()

        validation_reports.extend(analyze_edge_types(nodes_df, edges_df, unique_edge_ids))

        return validation_reports


def run_config_range_checks(df: pl.DataFrame, config_contents):
    """Run the config range checks for both nodes or edges."""
    validation_reports = []
    column_names = df.collect_schema().names()
    logger.debug(f"{column_names}")

    if config_contents:
        if "nodes_attribute_checks" in config_contents:
            for check in config_contents["nodes_attribute_checks"]["checks"]:
                if "range" in check:
                    range_check = check["range"]
                    if range_check["column"] in column_names:
                        validation_reports.append(
                            check_column_range(df, range_check["column"], int(range_check["min"]), int(range_check["max"]))
                        )
                if "enum" in check:
                    enum_check = check["enum"]
                    if enum_check["column"] in column_names:
                        validation_reports.append(check_column_enum(df, enum_check["column"], list(enum_check["values"])))

        if "edges_attribute_checks" in config_contents:
            for check in config_contents["edges_attribute_checks"]["checks"]:
                if "range" in check:
                    if check["range"]["column"] in column_names:
                        validation_reports.append(
                            check_column_range(df, check["range"]["column"], int(check["range"]["min"]), int(check["range"]["max"]))
                        )
                if "enum" in check:
                    if check["enum"]["column"] in column_names:
                        validation_reports.append(check_column_enum(df, check["range"]["column"], list(check["range"]["values"])))

    return validation_reports


def analyze_edge_types(nodes_df: pl.DataFrame, edges_df: pl.DataFrame, unique_edge_ids):
    """Analyzing Edge types."""
    validation_reports = []

    # Check if all edge IDs exist in node IDs
    counts_df = nodes_df.select([(~pl.col("id").str.contains_any(unique_edge_ids)).sum().alias("invalid_edge_ids_in_node_ids_count")])

    logger.info(counts_df.head())

    # Check if there are missing node IDs referenced in edges
    if counts_df.get_column("invalid_edge_ids_in_node_ids_count").item(0) > 0:
        validation_reports.append(check_edge_ids_in_node_ids(nodes_df, unique_edge_ids, "id"))

    # Analyze edge types
    logger.info("Analyzing edge types")
    try:
        # Log the DataFrame structures for debugging
        logger.debug(f"Nodes DataFrame columns: {nodes_df.columns}")
        logger.debug(f"Edges DataFrame columns: {edges_df.columns}")

        edge_type_analysis = util.analyze_edge_types(nodes_df, edges_df)
        logger.info(f"Edge type analysis results shape: {edge_type_analysis.shape}")

        # Get counts of valid and invalid edge types
        valid_edges = edge_type_analysis.filter(pl.col("valid"))
        invalid_edges = edge_type_analysis.filter(~pl.col("valid"))

        # Count unique edge types and edge instances
        unique_valid_types = len(valid_edges)
        unique_invalid_types = len(invalid_edges)
        total_unique_types = unique_valid_types + unique_invalid_types

        valid_count = valid_edges.select(pl.sum("count")).item(0, 0) if not valid_edges.is_empty() else 0
        invalid_count = invalid_edges.select(pl.sum("count")).item(0, 0) if not invalid_edges.is_empty() else 0
        total_count = valid_count + invalid_count

        logger.info(f"Valid edges: {valid_count} ({unique_valid_types} unique types)")
        logger.info(f"Invalid edges: {invalid_count} ({unique_invalid_types} unique types)")
        logger.info(f"Total: {total_count} ({total_unique_types} unique types)")

        # Calculate percentages
        valid_percent = (valid_count / total_count * 100) if total_count > 0 else 0
        invalid_percent = (invalid_count / total_count * 100) if total_count > 0 else 0

        # Create a JSON report for edge type analysis
        edge_type_summary = {
            "info": {
                "edge_type_analysis": {
                    "total_edges": total_count,
                    "valid_edges": {"count": valid_count, "percent": round(valid_percent, 2), "unique_types": unique_valid_types},
                    "invalid_edges": {"count": invalid_count, "percent": round(invalid_percent, 2), "unique_types": unique_invalid_types},
                }
            }
        }

        validation_reports.append(json.dumps(edge_type_summary))

        # Add more detailed report about invalid edge types if there are any
        if invalid_count > 0:
            invalid_edge_types = edge_type_analysis.filter(~pl.col("valid")).sort("count", descending=True)
            logger.info(f"Found {unique_invalid_types} distinct invalid edge types")

            # Format invalid edge types for reporting (limit to top 20 for readability)
            invalid_types_report = {}
            for row in invalid_edge_types.head(20).iter_rows(named=True):
                key = f"{row['subject_category']}-{row['predicate']}-{row['object_category']}"
                value = row["count"]
                invalid_types_report[key] = value

            # Build a dictionary for invalid edge types
            invalid_edge_types_dict = {
                "error": {
                    "message": f"{invalid_count} edges ({invalid_percent:.2f}%) have invalid edge types according to the biolink model.",
                    "valid": {"count": valid_count, "percent": round(valid_percent, 2), "unique_types": unique_valid_types},
                    "invalid": {
                        "count": invalid_count,
                        "percent": round(invalid_percent, 2),
                        "unique_types": unique_invalid_types,
                        "top_invalid_edge_types": invalid_types_report,
                    },
                }
            }
            validation_reports.append(json.dumps(invalid_edge_types_dict))
            logger.info("Added invalid edge types warning to validation report")
        else:
            success_message = {
                "info": {
                    "message": f"All {total_count} edges have valid edge types according to the biolink model."
                    f"Number of unique valid edge types: {unique_valid_types}"
                }
            }
            validation_reports.append(json.dumps(success_message))
            logger.info(success_message)

    except Exception as e:
        logger.error(f"Error analyzing edge types: {str(e)}")
        logger.exception("Full traceback:")
        # Don't add to validation reports as this is not a critical error

    return validation_reports


def check_for_superfluous_columns(source: str, schema: pt.Model, df: pl.DataFrame):
    """Find superfluous columns in a KG from a Polars Dataframe."""
    validation_reports = []
    try:
        schema.validate(df, allow_missing_columns=True, allow_superfluous_columns=False)
    except pt.exceptions.DataFrameValidationError as ex:
        superfluous_columns = []
        superfluous_columns.extend([_display_error_loc(e) for e in ex.errors() if e["type"] == "type_error.superfluouscolumns"])
        if superfluous_columns:
            violation = {
                "warning": {
                    "source": source,
                    "check": "superfluous_columns_not_recognized_by_biolink_model",
                    "columns": f"{','.join(superfluous_columns)}",
                }
            }
            validation_reports.append(json.dumps(violation))
    return validation_reports


def check_schema(source: str, schema: pt.Model, df: pl.DataFrame):
    """Validate a KG from a Polars Dataframe."""
    validation_reports = []
    try:
        schema.validate(df, allow_missing_columns=False, allow_superfluous_columns=True)
    except pt.exceptions.DataFrameValidationError as ex:
        columns_by_error_type_map = {}
        for e in ex.errors():
            type = e["type"].removeprefix("value_error.").removeprefix("type_error.")
            column = _display_error_loc(e)

            if type not in columns_by_error_type_map:
                columns_by_error_type_map[type] = set()

            if column not in columns_by_error_type_map[type]:
                columns_by_error_type_map[type].add(column)

        if columns_by_error_type_map:
            for key, value in columns_by_error_type_map.items():
                violation = {"error": {"source": source, "check": "schema_validation", "type": key, "column": value}}
                validation_reports.append(json.dumps(violation, default=serialize_sets))

    return validation_reports


def run_node_checks(df: pl.DataFrame, prefixes: list, class_prefix_map: dict):
    """Run Nodes Validation Checks using a Polars Dataframe."""
    usable_columns = [pl.col("id"), pl.col("category")]
    node_check_df = df.select(usable_columns)

    validation_reports = []
    try:
        logger.info("collecting node counts")

        counts_df = node_check_df.select(
            [
                (~pl.col("id").str.contains(CURIE_REGEX)).sum().alias("invalid_curie_id_count"),
                (~pl.col("id").str.contains_any(prefixes)).sum().alias("invalid_contains_biolink_model_prefix_id_count"),
                (~pl.col("category").str.contains(STARTS_WITH_BIOLINK_REGEX)).sum().alias("invalid_starts_with_biolink_category_count"),
                (~pl.col("category").str.contains(DELIMITED_BY_PIPES)).sum().alias("invalid_delimited_by_pipes_category_count"),
                (~pl.col("category").str.contains(NO_LEADING_WHITESPACE)).sum().alias("invalid_no_leading_whitespace_category_count"),
                (~pl.col("category").str.contains(NO_TRAILING_WHITESPACE)).sum().alias("invalid_no_trailing_whitespace_category_count"),
            ]
        )

        logger.debug(counts_df.head())

        if counts_df.get_column("invalid_curie_id_count").item(0) > 0:
            validation_reports.append(check_column_is_valid_curie(node_check_df, "id"))

        if counts_df.get_column("invalid_contains_biolink_model_prefix_id_count").item(0) > 0:
            validation_reports.append(check_column_contains_biolink_model_prefix(node_check_df, "id", prefixes))

        if counts_df.get_column("invalid_no_leading_whitespace_category_count").item(0) > 0:
            validation_reports.append(check_column_no_leading_whitespace(node_check_df, "category"))

        if counts_df.get_column("invalid_no_trailing_whitespace_category_count").item(0) > 0:
            validation_reports.append(check_column_no_trailing_whitespace(node_check_df, "category"))

        if counts_df.get_column("invalid_starts_with_biolink_category_count").item(0) > 0:
            validation_reports.append(check_column_starts_with_biolink(node_check_df, "category"))

        if counts_df.get_column("invalid_delimited_by_pipes_category_count").item(0) > 0:
            validation_reports.append(check_column_is_delimited_by_pipes(node_check_df, "category"))

        tmp_violation = check_node_id_and_category_with_biolink_preferred_prefixes(class_prefix_map, node_check_df)
        if tmp_violation:
            validation_reports.append(tmp_violation)

        logger.debug(f"number of total violations: {len(validation_reports)}")

    except pl.exceptions.ColumnNotFoundError as ex:
        violation = {
            "error": {
                "source": "nodes",
                "check": "Missing Required Column",
                "columns": f"{repr(ex)}",
            }
        }
        validation_reports.append(json.dumps(violation))
    except Exception as ex:
        logger.exception(ex)

    return validation_reports


def run_edge_checks(df: pl.DataFrame, prefixes: list):
    """Run Edge Validation Checks using a Polars Dataframe."""
    usable_columns = [pl.col("subject"), pl.col("predicate"), pl.col("object"), pl.col("knowledge_level"), pl.col("agent_type")]
    edge_checks_df = df.select(usable_columns)
    validation_reports = []
    try:
        logger.info("collecting edge counts")

        counts_df = edge_checks_df.select(
            [
                (~pl.col("subject").str.contains(CURIE_REGEX)).sum().alias("invalid_curie_subject_count"),
                (~pl.col("subject").str.contains_any(prefixes)).sum().alias("invalid_contains_biolink_model_prefix_subject_count"),
                (~pl.col("predicate").str.contains(STARTS_WITH_BIOLINK_REGEX)).sum().alias("invalid_starts_with_biolink_predicate_count"),
                (~pl.col("object").str.contains(CURIE_REGEX)).sum().alias("invalid_curie_object_count"),
                (~pl.col("object").str.contains_any(prefixes)).sum().alias("invalid_contains_biolink_model_prefix_object_count"),
                (~pl.col("knowledge_level").str.contains_any(BIOLINK_KNOWLEDGE_LEVEL_KEYS))
                .sum()
                .alias("invalid_contains_biolink_model_knowledge_level_count"),
                (~pl.col("agent_type").str.contains_any(BIOLINK_AGENT_TYPE_KEYS))
                .sum()
                .alias("invalid_contains_biolink_model_agent_type_count"),
            ]
        )

        logger.info(counts_df.head())

        if counts_df.get_column("invalid_curie_subject_count").item(0) > 0:
            validation_reports.append(check_column_is_valid_curie(edge_checks_df, "subject"))

        if counts_df.get_column("invalid_contains_biolink_model_prefix_subject_count").item(0) > 0:
            validation_reports.append(check_column_contains_biolink_model_prefix(edge_checks_df, "subject", prefixes))

        if counts_df.get_column("invalid_curie_object_count").item(0) > 0:
            validation_reports.append(check_column_is_valid_curie(edge_checks_df, "object"))

        if counts_df.get_column("invalid_contains_biolink_model_prefix_object_count").item(0) > 0:
            validation_reports.append(check_column_contains_biolink_model_prefix(edge_checks_df, "object", prefixes))

        if counts_df.get_column("invalid_starts_with_biolink_predicate_count").item(0) > 0:
            validation_reports.append(check_column_starts_with_biolink(edge_checks_df, "predicate"))

        if counts_df.get_column("invalid_contains_biolink_model_knowledge_level_count").item(0) > 0:
            validation_reports.append(
                check_column_contains_biolink_model_knowledge_level(edge_checks_df, "knowledge_level", BIOLINK_KNOWLEDGE_LEVEL_KEYS)
            )

        if counts_df.get_column("invalid_contains_biolink_model_agent_type_count").item(0) > 0:
            validation_reports.append(check_column_contains_biolink_model_agent_type(edge_checks_df, "agent_type", BIOLINK_AGENT_TYPE_KEYS))

    except pl.exceptions.ColumnNotFoundError as ex:
        logger.error(f"missing required column: {repr(ex)}")
        violation = {
            "error": {
                "source": "edges",
                "check": "Missing Required Column",
                "columns": f"{repr(ex)}",
            }
        }
        validation_reports.append(json.dumps(violation))
    return validation_reports
