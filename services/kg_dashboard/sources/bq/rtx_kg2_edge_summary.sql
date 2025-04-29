WITH summary AS (
  SELECT 'Ingested' AS name, COUNT(*) AS value, 1 as sort_order
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_edges_ingested_${rtx_kg2_version}`

  UNION ALL

  SELECT 'Transformed' AS name, COUNT(*) AS value, 2 as sort_order
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_edges_transformed`

  UNION ALL

  SELECT 'Normalized' AS name, COUNT(*) AS value, 3 as sort_order
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_edges_normalized`
)

SELECT name, value, sort_order
FROM summary

