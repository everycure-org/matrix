SELECT
  COALESCE(ec_core_category, 'other') as category,
  is_ontology_connected,
  COUNT(*) as node_count
FROM `${project_id}.release_${bq_release_version}.node_metrics`
GROUP BY category, is_ontology_connected
