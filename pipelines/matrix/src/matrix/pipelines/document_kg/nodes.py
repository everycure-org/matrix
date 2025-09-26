import logging
from typing import Any, Dict, List, Optional

import pandas
import pyspark.sql as ps
import pyspark.sql.functions as F
from jinja2 import Template

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_pks_from_unified_edges(unified_edges: ps.DataFrame) -> pandas.DataFrame:
    """Extract all unique primary knowledge sources from the unified edges data.

    Args:
        unified_edges (ps.DataFrame): The unified edges Spark DataFrame containing PKS information
                                     in columns: primary_knowledge_source, primary_knowledge_sources,
                                     aggregator_knowledge_source

    Returns:
        pandas.DataFrame: DataFrame with single column 'primary_knowledge_source' containing
                         all unique PKS identifiers found in the edges
    """
    # Extract from single-value primary_knowledge_source column
    single_pks = (
        unified_edges.select("primary_knowledge_source")
        .filter(F.col("primary_knowledge_source").isNotNull())
        .select(F.col("primary_knowledge_source").alias("pks"))
    )

    # Extract from array-valued primary_knowledge_sources column
    array_pks = unified_edges.select(F.explode("primary_knowledge_sources").alias("pks")).filter(
        F.col("pks").isNotNull()
    )

    # Extract from array-valued aggregator_knowledge_source column
    agg_pks = unified_edges.select(F.explode("aggregator_knowledge_source").alias("pks")).filter(
        F.col("pks").isNotNull()
    )

    # Union all PKS sources and get unique values
    unique_pks = single_pks.union(array_pks).union(agg_pks).distinct()

    # Convert to pandas DataFrame with expected column name
    pks_pandas = unique_pks.select(F.col("pks").alias("primary_knowledge_source")).toPandas()

    return pks_pandas


## Parsing and combining the PKS metadata source files


def _parse_source(
    source_data: List[Dict[str, Any]],
    source_id: str,
    id_column: str,
    extracted_metadata: List[str],
    ignored_metadata: List[str],
    primary_knowledge_sources: Dict[str, Dict[str, Any]],
) -> None:
    """Parse a single source of PKS metadata and integrate it into the main dictionary."""

    potentially_useful_keys = set()

    for record in source_data:
        raw_id = record[id_column]
        id = raw_id.replace("infores:", "")
        if id not in primary_knowledge_sources:
            primary_knowledge_sources[id] = {}

        for key in record:
            if key not in extracted_metadata + ignored_metadata + [id_column]:
                potentially_useful_keys.add(key)

        data_extract = {}
        data_extract[id_column] = raw_id
        for element in extracted_metadata:
            if element in record:
                data_extract[element] = record[element]

        primary_knowledge_sources[id][source_id] = data_extract

    if len(potentially_useful_keys) > 0:
        logger.warning(
            f"Found potentially useful keys in {source_id} that are not extracted: {potentially_useful_keys}"
        )


def _apply_infores_mapping(
    mapping: Optional[Dict[str, str]], data_to_map: List[Dict[str, Any]], id_column: str
) -> None:
    """Apply mapping from source IDs to the canonical infores IDs."""
    if mapping:
        for record in data_to_map:
            record["updated_id"] = mapping.get(record[id_column], record[id_column])


def _parse_infores(infores_d: Dict[str, Any], primary_knowledge_sources: Dict[str, Dict[str, Any]]) -> None:
    """Parse the infores metadata and integrate it into the primary knowledge sources."""
    source = "infores"
    id_column = "id"
    ignored_metadata = []
    extracted_metadata = [
        "id",
        "status",
        "name",
        "description",
        "knowledge_level",
        "agent_type",
        "url",
        "xref",
        "synonym",
        "consumed_by",
        "consumes",
    ]
    _parse_source(
        infores_d["information_resources"],
        source,
        id_column,
        extracted_metadata,
        ignored_metadata,
        primary_knowledge_sources,
    )


def _parse_reusabledata(
    reusabledata_d: List[Dict[str, Any]],
    primary_knowledge_sources: Dict[str, Dict[str, Any]],
    infores_mapping: Optional[Dict[str, str]],
) -> None:
    """Parse the reusabledata.org metadata and integrate it into the primary knowledge sources."""
    source = "reusabledata"
    id_column = "updated_id"
    ignored_metadata = ["last-curated", "grants"]
    extracted_metadata = [
        "id",
        "description",
        "source",
        "data-tags",
        "grade-automatic",
        "source-link",
        "source-type",
        "status",
        "data-field",
        "data-type",
        "data-categories",
        "data-access",
        "license",
        "license-type",
        "license-link",
        "license-hat-used",
        "license-issues",
        "license-commentary",
        "license-commentary-embeddable",
        "was-controversial",
        "provisional",
        "contacts",
    ]
    _apply_infores_mapping(infores_mapping, reusabledata_d, id_column="id")
    _parse_source(reusabledata_d, source, id_column, extracted_metadata, ignored_metadata, primary_knowledge_sources)


def _parse_kgregistry(
    kgregistry_d: Dict[str, Any],
    primary_knowledge_sources: Dict[str, Dict[str, Any]],
    infores_mapping: Optional[Dict[str, str]],
) -> None:
    """Parse the kgregistry metadata and integrate it into the primary knowledge sources."""
    source = "kgregistry"
    id_column = "updated_id"
    ignored_metadata = ["products"]
    extracted_metadata = [
        "id",
        "activity_status",
        "category",
        "collection",
        "contacts",
        "creation_date",
        "curators",
        "description",
        "domains",
        "evaluation_page",
        "fairsharing_id",
        "homepage_url",
        "infores_id",
        "language",
        "last_modified_date",
        "layout",
        "license",
        "name",
        "publications",
        "repository",
        "tags",
        "usages",
        "version",
        "warnings",
    ]
    kgregistry_data = kgregistry_d["resources"]
    _apply_infores_mapping(infores_mapping, kgregistry_data, "id")

    # Since we already have a few infores ids in the kgregistry, we should use them, even if we dont have an explicit mapping
    for record in kgregistry_data:
        record["updated_id"] = record.get("infores_id", record["id"])
    _parse_source(kgregistry_data, source, id_column, extracted_metadata, ignored_metadata, primary_knowledge_sources)


def _parse_matrixcurated(
    matrixcurated_d: pandas.DataFrame, primary_knowledge_sources: Dict[str, Dict[str, Any]]
) -> None:
    """Parse the matrixcurated metadata (mostly licensing information) and integrate it into the primary knowledge sources."""
    source = "matrixcurated"
    id_column = "primary_knowledge_source"
    ignored_metadata = ["aggregator_knowledge_source", "number_of_edges", "infores_name", "xref"]
    extracted_metadata = ["license_name", "license_source_link"]
    _parse_source(
        matrixcurated_d.to_dict(orient="records"),
        source,
        id_column,
        extracted_metadata,
        ignored_metadata,
        primary_knowledge_sources,
    )


def _parse_matrixreviews(
    matrixreviews_d: pandas.DataFrame, primary_knowledge_sources: Dict[str, Dict[str, Any]]
) -> None:
    """Parse the manually curated pks reviews according to the rubric."""
    source = "matrixreviews"
    id_column = "primary_knowledge_source"
    ignored_metadata = ["infores_name"]
    extracted_metadata = [
        "domain_coverage_score",
        "domain_coverage_comments",
        "source_scope_score",
        "source_scope_score_comment",
        "utility_drugrepurposing_score",
        "utility_drugrepurposing_comment",
        "label_rubric",
        "label_rubric_rationale",
        "label_manual",
        "label_manual_comment",
        "reviewer",
    ]
    _parse_source(
        matrixreviews_d.to_dict(orient="records"),
        source,
        id_column,
        extracted_metadata,
        ignored_metadata,
        primary_knowledge_sources,
    )


def _create_default_pks_entry(source_id: str) -> Dict[str, Any]:
    """Create a default entry for a PKS that has no metadata available."""
    return {
        "infores": {
            "id": f"infores:{source_id}",
            "name": f"{source_id} (no metadata available)",
            "description": "No metadata information was found for this primary knowledge source in any of the external registries.",
            "status": "unknown",
            "knowledge_level": "unknown",
            "agent_type": "unknown",
            "url": None,
            "xref": None,
            "synonym": None,
            "consumed_by": None,
            "consumes": None,
        }
    }


def _create_pks_subset_relevant_to_matrix(
    primary_knowledge_sources: Dict[str, Dict[str, Any]], relevant_sources: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Create a subset of the primary knowledge sources relevant to the Matrix project.

    For PKS that are not found in the metadata, default entries are created to ensure
    the pipeline doesn't fail and clearly indicates missing information.
    """
    subset = {}
    missing_sources = []

    for source in relevant_sources:
        if source in primary_knowledge_sources:
            subset[source] = primary_knowledge_sources[source]
        else:
            # Create default entry for missing PKS
            subset[source] = _create_default_pks_entry(source)
            missing_sources.append(source)

    if missing_sources:
        logger.warning(f"Created default entries for {len(missing_sources)} PKS with no metadata: {missing_sources}")

    return subset


def _parse_raw_data_for_pks_metadata(
    infores: Dict[str, Any],
    reusabledata: Dict[str, Any],
    kgregistry: Dict[str, Any],
    matrix_curated: pandas.DataFrame,
    matrix_reviews: pandas.DataFrame,
    mapping_reusabledata_infores: pandas.DataFrame,
    mapping_kgregistry_infores: pandas.DataFrame,
) -> Dict[str, Any]:
    """Parse the raw data for PKS metadata and integrate it into the primary knowledge sources.

    Args:
        infores (Dict[str, Any]): The infores metadata.
        reusabledata (Dict[str, Any]): The reusabledata.org metadata.
        kgregistry (Dict[str, Any]): The kgregistry metadata.
        matrix_curated (pandas.DataFrame): The matrix curated metadata.
        matrix_reviews (pandas.DataFrame): The matrix reviews metadata.
        mapping_reusabledata_infores (pandas.DataFrame): The mapping from reusabledata to infores.
        mapping_kgregistry_infores (pandas.DataFrame): The mapping from kgregistry to infores.

    Returns:
        dict[str, Any]: Integrated metadata from all PKS sources.
    """
    primary_knowledge_sources = {}
    _parse_infores(infores, primary_knowledge_sources)
    reusabledata_mapping_dict = _sssom_to_mapping_dict(mapping_reusabledata_infores)
    kgregistry_mapping_dict = _sssom_to_mapping_dict(mapping_kgregistry_infores)
    _parse_kgregistry(kgregistry, primary_knowledge_sources, kgregistry_mapping_dict)
    _parse_reusabledata(reusabledata, primary_knowledge_sources, reusabledata_mapping_dict)
    _parse_matrixcurated(matrix_curated, primary_knowledge_sources)
    _parse_matrixreviews(matrix_reviews, primary_knowledge_sources)

    return primary_knowledge_sources


## Generating the PKS documentation


def _get_property(source_info: Dict[str, Dict[str, Any]], property: str, default_value: str = "Unknown") -> str:
    """Get a property from the source info, checking multiple sources in order of priority."""
    property_value = default_value
    if "infores" in source_info and property in source_info["infores"]:
        property_value = source_info["infores"][property]
    elif "kgregistry" in source_info and property in source_info["kgregistry"]:
        property_value = source_info["kgregistry"][property]
    elif "reusabledata" in source_info and property in source_info["reusabledata"]:
        property_value = source_info["reusabledata"][property]
    return property_value


def _get_property_from_source(source_info: Dict[str, Dict[str, Any]], source: str, property: str) -> Optional[Any]:
    """Get a property from a specific source in the source info."""
    if source in source_info:
        if property in source_info[source]:
            value = source_info[source][property]
            if isinstance(value, str):
                return value.strip()
            else:
                return value
    return None


def _is_default_pks_entry(source_info: Dict[str, Dict[str, Any]]) -> bool:
    """Check if this is a default PKS entry (no metadata available)."""
    return (
        len(source_info) == 1
        and "infores" in source_info
        and source_info["infores"].get("description")
        == "No metadata information was found for this primary knowledge source in any of the external registries."
    )


def _format_license(source_info: Dict[str, Dict[str, Any]]) -> str:
    """Format the license information for a source into markdown."""

    # Handle default entries with no metadata
    if _is_default_pks_entry(source_info):
        return """#### License information
**No license information available** - This primary knowledge source was found in the knowledge graph but no metadata could be located in external registries.
"""

    pks_jinja2_template = Template("""#### License information
- **Matrix manual curation**: {%if matrix_license_name is not none %}[{{ matrix_license_name }}]({{ matrix_license_url }}){% else %}No license information curated.{% endif %}{%if kg_registry_license_id is not none %}
- **KG Registry**: {%if kg_registry_license_name is not none %}[{{ kg_registry_license_name }}]({{ kg_registry_license_id }}){% else %}[{{ kg_registry_license_id }}]({{ kg_registry_license_id }}){% endif %}{% endif %}{%if reusabledata_license is not none %}
- **Reusable Data**: {{ reusabledata_license }} ({{ reusabledata_license_type | default("Unknown license type") }}){% endif %}{%if reusabledata_license_issues is not none %}
    - _Issues_: {{ reusabledata_license_issues }}{% endif %}{%if reusabledata_license_commentary is not none %}
    - _Commentary_: {{ reusabledata_license_commentary }}{% endif %}
""")
    matrix_license_name = _get_property_from_source(source_info, "matrixcurated", "license_name")
    matrix_license_url = _get_property_from_source(source_info, "matrixcurated", "license_source_link")

    kgregistry_license = _get_property_from_source(source_info, "kgregistry", "license")
    kg_registry_license_name = (
        kgregistry_license["label"] if kgregistry_license is not None and "label" in kgregistry_license else None
    )
    kg_registry_license_id = (
        kgregistry_license["id"] if kgregistry_license is not None and "id" in kgregistry_license else None
    )

    reusabledata_license = _get_property_from_source(source_info, "reusabledata", "license")
    reusabledata_license_commentary = _get_property_from_source(
        source_info, "reusabledata", "license-commentary-embeddable"
    )
    reusabledata_license_issues = _get_property_from_source(source_info, "reusabledata", "license-issues")
    reusabledata_license_issues_string = None
    reusabledata_license_issues_list = []
    if reusabledata_license_issues is not None:
        if isinstance(reusabledata_license_issues, list):
            for issue in reusabledata_license_issues:
                issue_str = f"{issue['comment']} ({issue['criteria']})"
                reusabledata_license_issues_list.append(issue_str)
    if len(reusabledata_license_issues_list) > 0:
        reusabledata_license_issues_string = "; ".join(reusabledata_license_issues_list)
    reusabledata_license_type = _get_property_from_source(source_info, "reusabledata", "license-type")

    return pks_jinja2_template.render(
        matrix_license_name=matrix_license_name,
        matrix_license_url=matrix_license_url,
        kg_registry_license_name=kg_registry_license_name,
        kg_registry_license_id=kg_registry_license_id,
        reusabledata_license=reusabledata_license,
        reusabledata_license_type=reusabledata_license_type,
        reusabledata_license_issues=reusabledata_license_issues_string,
        reusabledata_license_commentary=reusabledata_license_commentary,
    )


def _format_review(source_info: Dict[str, Dict[str, Any]]) -> str:
    """Format the review information for a source into markdown."""

    # Handle default entries with no metadata
    if _is_default_pks_entry(source_info):
        return """#### Review information for this resource

   **No review information available** - This primary knowledge source was found in the knowledge graph but no metadata could be located in external registries.
"""

    pks_jinja2_template = Template("""#### Review information for this resource

{%if label_rubric is not none %}

??? note "Expand to see detailed review"

    Review information was generated specifically for the Matrix project and may not reflect the views of the broader community.

    - **Reviewer**: {{ reviewer }}
    - **Overall review score**:
        - Reviewer: `{{ label_manual }}` - {{ label_manual_comment }}
        - Rubric: `{{ label_rubric }}` - {{ label_rubric_rationale }}
    - **Domain Coverage**: `{{ domain_coverage_score }}` - {{ domain_coverage_comments }}
    - **Source Scope**: `{{ source_scope_score }}` - {{ source_scope_score_comment }}
    - **Drug Repurposing Utility**: `{{ utility_drugrepurposing_score }}` - {{ utility_drugrepurposing_comment }}
{% else %}
No review information available.
{% endif %}
""")
    domain_coverage_comments = _get_property_from_source(source_info, "matrixreviews", "domain_coverage_comments")
    domain_coverage_score = _get_property_from_source(source_info, "matrixreviews", "domain_coverage_score")
    label_manual = _get_property_from_source(source_info, "matrixreviews", "label_manual")
    label_manual_comment = _get_property_from_source(source_info, "matrixreviews", "label_manual_comment")
    label_rubric = _get_property_from_source(source_info, "matrixreviews", "label_rubric")
    label_rubric_rationale = _get_property_from_source(source_info, "matrixreviews", "label_rubric_rationale")
    reviewer = _get_property_from_source(source_info, "matrixreviews", "reviewer")
    source_scope_score = _get_property_from_source(source_info, "matrixreviews", "source_scope_score")
    source_scope_score_comment = _get_property_from_source(source_info, "matrixreviews", "source_scope_score_comment")
    utility_drugrepurposing_comment = _get_property_from_source(
        source_info, "matrixreviews", "utility_drugrepurposing_comment"
    )
    utility_drugrepurposing_score = _get_property_from_source(
        source_info, "matrixreviews", "utility_drugrepurposing_score"
    )

    return pks_jinja2_template.render(
        domain_coverage_comments=domain_coverage_comments,
        domain_coverage_score=domain_coverage_score,
        label_manual=label_manual,
        label_manual_comment=label_manual_comment,
        label_rubric=label_rubric,
        label_rubric_rationale=label_rubric_rationale,
        reviewer=reviewer,
        source_scope_score=source_scope_score,
        source_scope_score_comment=source_scope_score_comment,
        utility_drugrepurposing_comment=utility_drugrepurposing_comment,
        utility_drugrepurposing_score=utility_drugrepurposing_score,
    )


def _generate_list_of_pks_markdown_strings(source_data: Dict[str, Dict[str, Any]]) -> List[str]:
    """Generate a list of markdown strings for each PKS in the source data."""
    pks_jinja2_template = Template("""### Source: {{ title }} ({{ id }})

_{{ description }}_

{% if urls %}
**Links**:

{% for url in urls -%}
- [{{ url }}]({{ url }})
{% endfor %}{% endif %}

{{ license }}

{{ review }}""")

    pks_documentation_texts = []
    for source_id, source_info in source_data.items():
        name = _get_property(source_info, "name", default_value="No name")
        description = _get_property(source_info, "description", default_value="No description.")
        license = _format_license(source_info)
        review = _format_review(source_info)
        urls = []
        infores_url = _get_property_from_source(source_info, "infores", "xref")
        kgregistry_url = _get_property_from_source(source_info, "kgregistry", "homepage_url")
        reusabledata_url = _get_property_from_source(source_info, "reusabledata", "source-link")
        if infores_url:
            urls.extend(infores_url)
        if kgregistry_url:
            urls.append(kgregistry_url)
        if reusabledata_url:
            urls.append(reusabledata_url)
        urls.append(f"https://w3id.org/information-resource-registry/{source_id}")

        urls = list(set(urls))
        urls = sorted(urls)

        pks_docstring = pks_jinja2_template.render(
            id=source_id, title=name, description=description, urls=urls, license=license, review=review
        )
        pks_documentation_texts.append(pks_docstring)
    return pks_documentation_texts


def _generate_pks_markdown_documentation(pks_documentation_texts: List[str], overview_table: str) -> str:
    """Generate the full markdown documentation for the PKS."""
    pks_jinja2_template = Template("""# {{ title }}
                                   
This page is automatically generated with curated information about primary knowledge sources
leveraged in the MATRIX Knowledge Graph, mainly regarding licensing information and 
potential relevancy assessments for drug repurposing.

This internally curated information is augmented with information from three external resources:

1. [Information Resource Registry](https://biolink.github.io/information-resource-registry/)
2. [reusabledata.org](https://reusabledata.org/)
3. [KG Registry](https://kghub.org/kg-registry/)

## Overview

{{ overview_table }}

## Detailed information about each primary knowledge sources

{% for doc in pks_documentation_texts %}
{{ doc }}
{% endfor %}
""")
    pks_docs = pks_jinja2_template.render(
        title="KG Primary Knowledge Sources",
        pks_documentation_texts=pks_documentation_texts,
        overview_table=overview_table,
    )
    return pks_docs


def _generate_overview_table_of_pks_markdown(source_data: Dict[str, Dict[str, Any]]) -> str:
    """Generate an overview table of PKS in markdown format."""
    pks_jinja2_template = Template("""**Overview table**


{% if data %}
| Resource | License |
| -------- | ------- |
{% for rec in data -%}
| {{ rec.name }} | {%if rec.license_name is not none %}[{{ rec.license_name }}]({{ rec.license_url }}){% else %}No license information curated.{% endif %} |
{% endfor %}{% endif %}
""")

    license_data = []
    for source_id, source_info in source_data.items():
        name = _get_property(source_info, "name", default_value="No name")
        if name == "No name":
            continue
        matrix_license_name = _get_property_from_source(source_info, "matrixcurated", "license_name")
        matrix_license_url = _get_property_from_source(source_info, "matrixcurated", "license_source_link")
        rec = {"id": source_id, "name": name, "license_name": matrix_license_name, "license_url": matrix_license_url}
        license_data.append(rec)

    pks_table_docstring = pks_jinja2_template.render(
        data=license_data,
    )
    return pks_table_docstring


def _sssom_to_mapping_dict(sssom_data: pandas.DataFrame) -> Dict[str, str]:
    """Convert SSSOM mapping DataFrame to a simple key-value dictionary."""
    mapping_dict = {row["subject_id"]: row["object_id"] for _, row in sssom_data.iterrows()}
    return mapping_dict


def get_relevant_pks_ids(pks_integrated: pandas.DataFrame) -> List[str]:
    """Get unique primary knowledge source IDs from the integrated KG data.

    Args:
        pks_integrated (pandas.DataFrame): Table of PKS IDs present in the latest integrated KG.

    Returns:
        List[str]: List of unique PKS/infores IDs relevant to the integrated KG.
    """
    relevant_sources = [
        src.replace("infores:", "") for src in pks_integrated["primary_knowledge_source"].unique().tolist()
    ]
    return list(set(relevant_sources))


def create_pks_documentation(matrix_subset_relevant_sources: Dict[str, Any]) -> str:
    """Create markdown documentation for primary knowledge sources (PKS) used in the matrix.

    Args:
        matrix_subset_relevant_sources (Dict[str, Any]): Integrated metadata for PKS relevant to the provided matrix, keyed by PKS/infores ID.

    Returns:
        str: Markdown documentation string for the PKS used in the matrix.
    """
    pks_documentation_texts = _generate_list_of_pks_markdown_strings(matrix_subset_relevant_sources)
    overview_table = _generate_overview_table_of_pks_markdown(source_data=matrix_subset_relevant_sources)
    documentation_md = _generate_pks_markdown_documentation(pks_documentation_texts, overview_table)
    return documentation_md


def create_pks_integrated_metadata(
    infores: Dict[str, Any],
    reusabledata: Dict[str, Any],
    kgregistry: Dict[str, Any],
    matrix_curated: pandas.DataFrame,
    matrix_reviews: pandas.DataFrame,
    pks_integrated: pandas.DataFrame,
    mapping_reusabledata_infores: pandas.DataFrame,
    mapping_kgregistry_infores: pandas.DataFrame,
) -> Dict[str, Any]:
    """Create integrated metadata for primary knowledge source (PKS) IDs used in the matrix.

    Args:
        infores (dict[str, Any]): The full Information Resource Registry (https://biolink.github.io/information-resource-registry/).
        reusabledata (dict[str, Any]): Raw metadata from https://reusabledata.org/.
        kgregistry (dict[str, Any]): Raw metadata from https://kghub.org/kg-registry/
        matrix_curated (pandas.DataFrame): Matrix-curated PKS metadata.
        matrix_reviews (pandas.DataFrame): Matrix-curated relevancy for drug-repurposing assessments of PKS.
        pks_integrated (pandas.DataFrame): Table of PKS IDs present in the latest integrated KG. This is used to subset the full metadata.
        mapping_reusabledata_infores (pandas.DataFrame): Mapping from reusabledata entries to infores identifiers (manually curated by Matrix project).
        mapping_kgregistry_infores (pandas.DataFrame): Mapping from kgregistry entries to infores identifiers (manually curated by Matrix project).

    Returns:
        dict[str, Any]: Integrated metadata for PKS relevant to the provided matrix, keyed by PKS/infores ID.
    """
    relevant_sources = get_relevant_pks_ids(pks_integrated)
    primary_knowledge_sources = _parse_raw_data_for_pks_metadata(
        infores=infores,
        reusabledata=reusabledata,
        kgregistry=kgregistry,
        matrix_curated=matrix_curated,
        matrix_reviews=matrix_reviews,
        mapping_reusabledata_infores=mapping_reusabledata_infores,
        mapping_kgregistry_infores=mapping_kgregistry_infores,
    )
    matrix_subset_relevant_sources = _create_pks_subset_relevant_to_matrix(primary_knowledge_sources, relevant_sources)
    return matrix_subset_relevant_sources
