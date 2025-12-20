-- Aggregate metrics for Knowledge Graph index page comparison table
WITH kg_list AS (
  SELECT DISTINCT
    upstream_data_source as knowledge_graph,
    CASE upstream_data_source
      WHEN 'rtx_kg2' THEN 'RTX-KG2'
      WHEN 'robokop' THEN 'ROBOKOP'
      WHEN 'primekg' THEN 'PrimeKG'
      ELSE INITCAP(REPLACE(upstream_data_source, '_', ' '))
    END as display_name
  FROM `${project_id}.release_${bq_release_version}.unified_normalization_summary`
  WHERE upstream_data_source NOT IN (
    'disease_list', 'drug_list', 'ec_ground_truth',
    'ec_clinical_trials', 'kgml_xdtd_ground_truth', 'off_label'
  )
),

-- Node counts and normalization rates from unified_normalization_summary
node_metrics AS (
  SELECT
    upstream_data_source as knowledge_graph,
    COUNT(DISTINCT id) as node_count,
    COUNTIF(normalization_success = true) as normalized_count,
    COUNT(*) as total_normalization_attempts,
    SAFE_DIVIDE(COUNTIF(normalization_success = true), COUNT(*)) as normalization_success_rate
  FROM `${project_id}.release_${bq_release_version}.unified_normalization_summary`
  WHERE upstream_data_source NOT IN (
    'disease_list', 'drug_list', 'ec_ground_truth',
    'ec_clinical_trials', 'kgml_xdtd_ground_truth', 'off_label'
  )
    AND id != "['Error']"
  GROUP BY upstream_data_source
),

-- Edge counts and quality metrics from edges_unified
edge_metrics AS (
  SELECT
    upstream_data_source.list[SAFE_OFFSET(0)].element as knowledge_graph,
    COUNT(*) as edge_count,
    SAFE_DIVIDE(COUNTIF(primary_knowledge_source IS NOT NULL AND TRIM(primary_knowledge_source) != ''), COUNT(*)) as provenance_rate,
    -- Knowledge level breakdown for quality indicator
    SAFE_DIVIDE(COUNTIF(LOWER(knowledge_level) = 'knowledge_assertion'), COUNT(*)) as knowledge_assertion_rate,
    SAFE_DIVIDE(COUNTIF(LOWER(agent_type) = 'manual_agent'), COUNT(*)) as manual_curation_rate
  FROM `${project_id}.release_${bq_release_version}.edges_unified`
  WHERE upstream_data_source.list[SAFE_OFFSET(0)].element NOT IN (
    'disease_list', 'drug_list', 'ec_ground_truth',
    'ec_clinical_trials', 'kgml_xdtd_ground_truth', 'off_label'
  )
  GROUP BY upstream_data_source.list[SAFE_OFFSET(0)].element
)

SELECT
  k.knowledge_graph,
  k.display_name,
  n.node_count,
  e.edge_count,
  n.normalization_success_rate,
  e.provenance_rate,
  e.knowledge_assertion_rate,
  e.manual_curation_rate
FROM kg_list k
LEFT JOIN node_metrics n ON k.knowledge_graph = n.knowledge_graph
LEFT JOIN edge_metrics e ON k.knowledge_graph = e.knowledge_graph
ORDER BY k.knowledge_graph
