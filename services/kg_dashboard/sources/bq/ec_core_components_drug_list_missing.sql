-- Extra elements in drug list not in nodes_unified
SELECT
  CASE
    WHEN nu.id IS NULL THEN 'Missing'
    ELSE 'Included'
  END AS status,
  COUNT(*) AS count,
  dl.id,
  dl.name
FROM
  `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dl
LEFT JOIN
  `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_unified` nu
  ON dl.id = nu.id
GROUP BY status, dl.id, dl.name
ORDER BY status, id