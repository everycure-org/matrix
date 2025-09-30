import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
from jinja2 import Template
from matrix_inject.inject import inject_object

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_pks_from_unified_edges(unified_edges: ps.DataFrame) -> List[str]:
    """Extract all unique primary knowledge sources from the unified edges data.

    Args:
        unified_edges (ps.DataFrame): The unified edges Spark DataFrame containing PKS information
                                     in columns: primary_knowledge_source, primary_knowledge_sources,
                                     aggregator_knowledge_source

    Returns:
        List[str]: List of unique PKS/infores IDs (with 'infores:' prefix stripped)
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

    # Convert to list, strip infores: prefix
    relevant_sources = [src.replace("infores:", "") for src in unique_pks.toPandas()["pks"].tolist()]

    return relevant_sources


@inject_object()
def parse_pks_source(
    parser: Callable,
    source_data: Any,
    config: Dict[str, Any],
    mapping_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generic parser dispatcher that calls source-specific parser function.

    Args:
        parser: Parser function from source_parsers module
        source_data: Raw data (format varies by source)
        config: Configuration from parameters.yml for this source
        mapping_data: Optional mapping DataFrame for sources that need ID mapping

    Returns:
        Dict keyed by PKS ID with extracted metadata for this source
    """
    # Call with or without mapping based on whether mapping_data was provided by pipeline
    if mapping_data is not None:
        return parser(source_data, mapping_data, config)
    else:
        return parser(source_data, config)


def merge_all_pks_metadata(*source_dicts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merge multiple PKS metadata dictionaries.

    Args:
        *source_dicts: Variable number of dictionary arguments

    Returns:
        Combined dictionary with all PKS metadata
    """
    result = {}

    # Loop through all provided source dictionaries
    for source_dict in source_dicts:
        for pks_id, pks_data in source_dict.items():
            # If pks_id not in result, initialize it as empty dict
            if pks_id not in result:
                result[pks_id] = {}
            # Update result[pks_id] with pks_data (merging metadata)
            result[pks_id].update(pks_data)

    # Log the count of unique PKS and number of sources merged
    logger.info(f"Merged metadata for {len(result)} unique PKS from {len(source_dicts)} sources")

    return result


def integrate_all_metadata(
    all_pks_metadata: Dict[str, Dict[str, Any]],
    unified_edges: ps.DataFrame,
) -> Dict[str, Any]:
    """Filter PKS metadata to only relevant sources found in unified edges.

    Args:
        all_pks_metadata: All PKS metadata from all sources
        unified_edges: Unified edges to extract relevant PKS IDs from

    Returns:
        Filtered metadata for PKS relevant to the matrix
    """
    relevant_sources = extract_pks_from_unified_edges(unified_edges)
    return _create_pks_subset_relevant_to_matrix(all_pks_metadata, relevant_sources)


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
