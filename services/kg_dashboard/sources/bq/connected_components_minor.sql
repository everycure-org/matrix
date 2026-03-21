SELECT
  component_id,
  component_size,
  num_drugs,
  num_diseases,
  num_other,
  component_hash
FROM `${project_id}.release_${bq_release_version}.connected_components`
WHERE component_id > 0
ORDER BY component_size DESC
LIMIT 20
