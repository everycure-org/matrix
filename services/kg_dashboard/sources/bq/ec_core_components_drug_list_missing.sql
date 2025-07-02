-- Extra elements in drug list not in nodes_unified
SELECT 
  dl.id,
  dl.name
FROM
  `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dl
LEFT JOIN
  `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_unified` nu
  ON dl.id = nu.id
WHERE
  nu.id IS NULL
