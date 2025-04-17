SELECT 'Ingested' AS name, COUNT(*) AS value
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_nodes_ingested_${rtx_kg2_version}`

UNION ALL

SELECT 'Transformed', COUNT(*) AS value
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_nodes_transformed`

UNION ALL

SELECT 'Normalized', COUNT(*) AS value
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_nodes_normalized`
