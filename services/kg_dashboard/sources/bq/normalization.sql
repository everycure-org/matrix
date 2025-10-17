SELECT
  SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
  SPLIT(id, ':')[OFFSET(0)] AS prefix,
  all_categories.list[SAFE_OFFSET(0)].element AS category,
  normalization_success,
  CASE WHEN original_id = id THEN true ELSE false END AS no_normalization_change,
  upstream_data_source AS normalization_set,
  COUNT(DISTINCT id) AS count
FROM
  `${project_id}.release_${bq_release_version}.unified_normalization_summary`
WHERE id != "['Error']"
GROUP BY ALL

UNION ALL

SELECT
  SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
  SPLIT(id, ':')[OFFSET(0)] AS prefix,
  all_categories.list[SAFE_OFFSET(0)].element AS category,
  normalization_success,
  CASE WHEN original_id = id THEN true ELSE false END AS no_normalization_change,
  'merged' AS normalization_set,
  COUNT(DISTINCT id) AS count
FROM
  `${project_id}.release_${bq_release_version}.unified_normalization_summary`
WHERE id != "['Error']"
GROUP BY ALL
