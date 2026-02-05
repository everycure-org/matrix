-- Top categories by edge count per Knowledge Graph
-- Source query: returns categories for all KGs; pages filter by knowledge_graph param
SELECT
  edges.upstream_data_source.list[SAFE_OFFSET(0)].element as knowledge_graph,
  subject_nodes.category as subject_category,
  object_nodes.category as object_category,
  CONCAT(
    REPLACE(subject_nodes.category, 'biolink:', ''),
    ' → ',
    REPLACE(object_nodes.category, 'biolink:', '')
  ) as category_pair,
  COUNT(*) as edge_count
FROM `${project_id}.release_${bq_release_version}.edges_unified` AS edges
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` AS subject_nodes ON edges.subject = subject_nodes.id
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` AS object_nodes ON edges.object = object_nodes.id
WHERE edges.upstream_data_source.list[SAFE_OFFSET(0)].element NOT IN (
  'disease_list', 'drug_list', 'ec_ground_truth',
  'ec_clinical_trials', 'kgml_xdtd_ground_truth', 'off_label'
)
GROUP BY edges.upstream_data_source.list[SAFE_OFFSET(0)].element, subject_nodes.category, object_nodes.category
ORDER BY knowledge_graph, edge_count DESC
