SELECT
  CASE
    WHEN nu.id IS NULL THEN 'Missing'
    ELSE 'Included'
  END AS status,
  COUNT(*) AS count,
  dl.id,
  dl.name
FROM
  `${project_id}.release_${bq_release_version}.disease_list_nodes_normalized` dl
LEFT JOIN
  `${project_id}.release_${bq_release_version}.nodes_unified` nu
  ON dl.id = nu.id
GROUP BY status, dl.id, dl.name
ORDER BY status, id