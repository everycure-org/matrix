SELECT
  normalization_success,
  'prefix' AS dimension,
  SPLIT(id, ':')[OFFSET(0)] AS name,
  COUNT(*) AS count
FROM `${project_id}.release_${bq_release_version}.disease_list_normalization_summary`
GROUP BY
  normalization_success,
  name
ORDER BY
  normalization_success DESC,
  count DESC;