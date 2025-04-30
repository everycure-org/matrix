SELECT
IF(normalization_success, 'Success', 'Failure') AS name,
  COUNT(*) AS value
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_normalization_summary`
WHERE upstream_data_source = 'disease_list'
GROUP BY name