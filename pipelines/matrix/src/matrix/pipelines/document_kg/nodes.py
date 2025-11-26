import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
from jinja2 import Template
from matrix_inject.inject import inject_object

logger = logging.getLogger(__name__)


def extract_pks_from_unified_edges(unified_edges: ps.DataFrame) -> List[str]:
    """Extract unique primary knowledge sources from unified edges."""

    single_pks = (
        unified_edges.select("primary_knowledge_source")
        .filter(F.col("primary_knowledge_source").isNotNull())
        .select(F.col("primary_knowledge_source").alias("pks"))
    )

    array_pks = unified_edges.select(F.explode("primary_knowledge_sources").alias("pks")).filter(
        F.col("pks").isNotNull()
    )

    agg_pks = unified_edges.select(F.explode("aggregator_knowledge_source").alias("pks")).filter(
        F.col("pks").isNotNull()
    )

    unique_pks = single_pks.union(array_pks).union(agg_pks).distinct()
    relevant_sources = [src.replace("infores:", "") for src in unique_pks.toPandas()["pks"].tolist()]

    return relevant_sources


@inject_object()
def parse_pks_source(
    parser,
    source_data: Any,
    mapping_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, Any]]:
    """Parse PKS source using parser instance."""
    result = parser.parse(source_data, mapping_data)
    logger.info(f"Parsed {len(result)} PKS entries from {parser.name}")
    return result


def merge_all_pks_metadata(*source_dicts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merge PKS metadata from multiple sources."""
    result = {}

    for idx, source_dict in enumerate(source_dicts):
        num_entries = len(source_dict) if source_dict else 0
        logger.info(f"Source dict {idx}: {num_entries} PKS entries")
        for pks_id, pks_data in source_dict.items():
            if pks_id not in result:
                result[pks_id] = {}
            result[pks_id].update(pks_data)

    logger.info(f"Merged metadata for {len(result)} unique PKS from {len(source_dicts)} sources")

    return result


def integrate_all_metadata(
    all_pks_metadata: Dict[str, Dict[str, Any]],
    unified_edges: ps.DataFrame,
) -> Dict[str, Any]:
    """Filter PKS metadata to sources found in unified edges."""
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
    """Create subset of PKS metadata, generating defaults for missing sources."""
    subset = {}
    missing_sources = []

    for source in relevant_sources:
        if source in primary_knowledge_sources:
            subset[source] = primary_knowledge_sources[source]
        else:
            subset[source] = _create_default_pks_entry(source)
            missing_sources.append(source)

    if missing_sources:
        logger.warning(f"Created default entries for {len(missing_sources)} PKS with no metadata: {missing_sources}")

    return subset


def _get_property(source_info: Dict[str, Dict[str, Any]], property: str, default_value: str = "Unknown") -> str:
    """Get property from source info, checking infores, kgregistry, then reusabledata."""
    property_value = default_value
    if "infores" in source_info and property in source_info["infores"]:
        property_value = source_info["infores"][property]
    elif "kgregistry" in source_info and property in source_info["kgregistry"]:
        property_value = source_info["kgregistry"][property]
    elif "reusabledata" in source_info and property in source_info["reusabledata"]:
        property_value = source_info["reusabledata"][property]
    return property_value


def _get_property_from_source(source_info: Dict[str, Dict[str, Any]], source: str, property: str) -> Optional[Any]:
    """Get property from specific source."""
    if source in source_info:
        if property in source_info[source]:
            value = source_info[source][property]
            if isinstance(value, str):
                return value.strip()
            else:
                return value
    return None


def _is_default_pks_entry(source_info: Dict[str, Dict[str, Any]]) -> bool:
    """Check if PKS entry has no metadata available."""
    return (
        len(source_info) == 1
        and "infores" in source_info
        and source_info["infores"].get("description")
        == "No metadata information was found for this primary knowledge source in any of the external registries."
    )


def _format_license_issues(license_issues: Optional[Any]) -> Optional[str]:
    """Format license issues list as semicolon-separated string."""
    if license_issues is None or not isinstance(license_issues, list):
        return None
    issue_strings = [f"{issue['comment']} ({issue['criteria']})" for issue in license_issues]
    return "; ".join(issue_strings) if issue_strings else None


def _format_license(source_info: Dict[str, Dict[str, Any]], template: dict) -> str:
    """Format license information as markdown."""
    if _is_default_pks_entry(source_info):
        return """#### License information
**No license information available** - This primary knowledge source was found in the knowledge graph but no metadata could be located in external registries.
"""

    pks_jinja2_template = Template(template["license"])
    return pks_jinja2_template.render(
        matrix_license_name=_get_property_from_source(source_info, "matrixcurated", "license_name"),
        matrix_license_url=_get_property_from_source(source_info, "matrixcurated", "license_source_link"),
        kg_registry_license_name=_get_property_from_source(source_info, "kgregistry", "license").get("label")
        if _get_property_from_source(source_info, "kgregistry", "license")
        else None,
        kg_registry_license_id=_get_property_from_source(source_info, "kgregistry", "license").get("id")
        if _get_property_from_source(source_info, "kgregistry", "license")
        else None,
        reusabledata_license=_get_property_from_source(source_info, "reusabledata", "license"),
        reusabledata_license_type=_get_property_from_source(source_info, "reusabledata", "license-type"),
        reusabledata_license_issues=_format_license_issues(
            _get_property_from_source(source_info, "reusabledata", "license-issues")
        ),
        reusabledata_license_commentary=_get_property_from_source(
            source_info, "reusabledata", "license-commentary-embeddable"
        ),
    )


def _format_review(source_info: Dict[str, Dict[str, Any]], template: dict) -> str:
    """Format review information as markdown."""

    if _is_default_pks_entry(source_info):
        return """#### Review information for this resource

   **No review information available** - This primary knowledge source was found in the knowledge graph but no metadata could be located in external registries.
"""

    pks_jinja2_template = Template(template["review"])
    return pks_jinja2_template.render(
        domain_coverage_comments=_get_property_from_source(source_info, "matrixreviews", "domain_coverage_comments"),
        domain_coverage_score=_get_property_from_source(source_info, "matrixreviews", "domain_coverage_score"),
        label_manual=_get_property_from_source(source_info, "matrixreviews", "label_manual"),
        label_manual_comment=_get_property_from_source(source_info, "matrixreviews", "label_manual_comment"),
        label_rubric=_get_property_from_source(source_info, "matrixreviews", "label_rubric"),
        label_rubric_rationale=_get_property_from_source(source_info, "matrixreviews", "label_rubric_rationale"),
        reviewer=_get_property_from_source(source_info, "matrixreviews", "reviewer"),
        source_scope_score=_get_property_from_source(source_info, "matrixreviews", "source_scope_score"),
        source_scope_score_comment=_get_property_from_source(
            source_info, "matrixreviews", "source_scope_score_comment"
        ),
        utility_drugrepurposing_comment=_get_property_from_source(
            source_info, "matrixreviews", "utility_drugrepurposing_comment"
        ),
        utility_drugrepurposing_score=_get_property_from_source(
            source_info, "matrixreviews", "utility_drugrepurposing_score"
        ),
    )


def _generate_list_of_pks_markdown_strings(source_data: Dict[str, Dict[str, Any]], template: dict) -> List[str]:
    """Generate markdown documentation for each PKS."""
    pks_jinja2_template = Template(template["pks_list"])

    pks_documentation_texts = []
    for source_id, source_info in source_data.items():
        name = _get_property(source_info, "name", default_value="No name")
        description = _get_property(source_info, "description", default_value="No description.")
        license = _format_license(source_info, template)
        review = _format_review(source_info, template)
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


def _generate_pks_markdown_documentation(
    pks_documentation_texts: List[str], overview_table: str, template: dict
) -> str:
    """Generate complete PKS markdown documentation."""
    pks_jinja2_template = Template(template["main"])
    pks_docs = pks_jinja2_template.render(
        title="KG Primary Knowledge Sources",
        pks_documentation_texts=pks_documentation_texts,
        overview_table=overview_table,
    )
    return pks_docs


def _generate_overview_table_of_pks_markdown(source_data: Dict[str, Dict[str, Any]], template: dict) -> str:
    """Generate PKS overview table in markdown."""
    pks_jinja2_template = Template(template["overview"])

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


def create_pks_documentation(matrix_subset_relevant_sources: Dict[str, Any], templates: Dict[str, str]) -> str:
    """Generate markdown documentation for PKS used in the matrix."""
    pks_documentation_texts = _generate_list_of_pks_markdown_strings(matrix_subset_relevant_sources, template=templates)
    overview_table = _generate_overview_table_of_pks_markdown(matrix_subset_relevant_sources, template=templates)
    documentation_md = _generate_pks_markdown_documentation(pks_documentation_texts, overview_table, template=templates)
    return documentation_md
