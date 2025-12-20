-- Top predicates by edge count per Knowledge Graph
-- Source query: returns predicates for all KGs; pages filter by knowledge_graph param
SELECT
  upstream_data_source.list[SAFE_OFFSET(0)].element as knowledge_graph,
  predicate,
  REPLACE(predicate, 'biolink:', '') as predicate_display,
  COUNT(*) as edge_count
FROM `${project_id}.release_${bq_release_version}.edges_unified`
WHERE upstream_data_source.list[SAFE_OFFSET(0)].element NOT IN (
  'disease_list', 'drug_list', 'ec_ground_truth',
  'ec_clinical_trials', 'kgml_xdtd_ground_truth', 'off_label'
)
GROUP BY upstream_data_source.list[SAFE_OFFSET(0)].element, predicate
ORDER BY knowledge_graph, edge_count DESC
