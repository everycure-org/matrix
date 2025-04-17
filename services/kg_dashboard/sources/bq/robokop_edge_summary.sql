SELECT 'Ingested' AS name, COUNT(*) AS value
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.robokop_edges_ingested_${robokop_version}`

UNION ALL

SELECT 'Transformed', COUNT(*) AS value
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.robokop_edges_transformed`

UNION ALL

SELECT 'Normalized', COUNT(*) AS value
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.robokop_edges_normalized`
