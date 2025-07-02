# Compare disease list with nodes_unified

-- Disease list normalization success breakdown by category
SELECT 
  CASE WHEN nu.id IS NULL THEN FALSE ELSE TRUE END AS normalization_success,
  dl.category AS name,
  COUNT(*) AS count,
  'category' AS dimension
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` dl
LEFT JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_unified` nu
  ON dl.id = nu.id
GROUP BY normalization_success, name;
