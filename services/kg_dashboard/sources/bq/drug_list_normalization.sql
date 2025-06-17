SELECT
IF(normalization_success, 'Success', 'Failure') AS name,
  COUNT(*) AS value
FROM `${project_id}.release_${bq_release_version}.drug_list_normalization_summary`
WHERE upstream_data_source = 'drug_list'
GROUP BY name