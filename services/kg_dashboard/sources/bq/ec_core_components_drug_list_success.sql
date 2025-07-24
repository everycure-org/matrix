# Compare durg list with nodes_unified

-- Drug list normalization success breakdown by category
SELECT 
  CASE WHEN nu.id IS NULL THEN FALSE ELSE TRUE END AS normalization_success,
  dl.category AS name,
  COUNT(*) AS count,
  'category' AS dimension
FROM `${project_id}.release_${bq_release_version}.drug_list_nodes_normalized` dl
LEFT JOIN `${project_id}.release_${bq_release_version}.nodes_unified` nu
  ON dl.id = nu.id
GROUP BY normalization_success, name;
