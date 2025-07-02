-- Extra elements in disease list not in nodes_unified
SELECT 
  dl.id,
  dl.name
FROM
  `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` dl
LEFT JOIN
  `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_unified` nu
  ON dl.id = nu.id
WHERE
  nu.id IS NULL
