from typing import Any, Dict

import pandas
from document_kg_utils import (create_pks_documentation,
                               create_pks_subset_relevant_to_matrix,
                               get_relevant_pks_ids,
                               parse_raw_data_for_pks_metadata)


def create_pks_integrated_metadata(
        infores: Dict[str, Any], 
        reusabledata: Dict[str, Any],
        kgregistry: Dict[str, Any],
        matrix_curated: pandas.DataFrame,
        matrix_reviews: pandas.DataFrame,
        pks_integrated: pandas.DataFrame,
        mapping_reusabledata_infores: pandas.DataFrame,
        mapping_kgregistry_infores: pandas.DataFrame
    ) -> tuple[Dict[str, Any], str]:
    """Create integrated metadata for primary knowledge source IDs."""
    
    relevant_sources = get_relevant_pks_ids(pks_integrated)
    primary_knowledge_sources = parse_raw_data_for_pks_metadata(
        infores=infores, 
        reusabledata=reusabledata,
        kgregistry=kgregistry,
        matrix_curated=matrix_curated,
        matrix_reviews=matrix_reviews,
        mapping_reusabledata_infores=mapping_reusabledata_infores,
        mapping_kgregistry_infores=mapping_kgregistry_infores
    )
    matrix_subset_relevant_sources = create_pks_subset_relevant_to_matrix(primary_knowledge_sources, relevant_sources)
    pks_markdown_documentation = create_pks_documentation(matrix_subset_relevant_sources)

    return matrix_subset_relevant_sources, pks_markdown_documentation

