WITH summary AS (
  SELECT 'Ingested' AS name, COUNT(*) AS value, 1 as sort_order
  FROM `${project_id}.release_${bq_release_version}.robokop_nodes_ingested_${robokop_version}`

  UNION ALL

  SELECT 'Transformed' AS name, COUNT(*) AS value, 2 as sort_order
  FROM `${project_id}.release_${bq_release_version}.robokop_nodes_transformed`

  UNION ALL

  SELECT 'Normalized' AS name, COUNT(*) AS value, 3 as sort_order
  FROM `${project_id}.release_${bq_release_version}.robokop_nodes_normalized`
)

SELECT name, value, sort_order
FROM summary

