SELECT
  normalization_success,
  'prefix' AS dimension,
  SPLIT(id, ':')[OFFSET(0)] AS name,
  COUNT(*) AS count
FROM `${project_id}.release_${bq_release_version}.drug_list_normalization_summary`
GROUP BY
  normalization_success,
  name

UNION ALL

SELECT
  normalization_success,
  'category' AS dimension,
  category AS name,
  COUNT(*) AS count
FROM `${project_id}.release_${bq_release_version}.drug_list_normalization_summary`
GROUP BY
  normalization_success,
  category

ORDER BY
  normalization_success DESC,
  dimension,
  count DESC;