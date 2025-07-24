# Compare disease list with nodes_unified

-- Normalization donut
SELECT 
  CASE WHEN nu.id IS NULL THEN 'Failure' ELSE 'Success' END AS name,
  COUNT(*) AS value
FROM `${project_id}.release_${bq_release_version}.disease_list_nodes_normalized` dl
LEFT JOIN `${project_id}.release_${bq_release_version}.nodes_unified` nu
  ON dl.id = nu.id
GROUP BY name;
