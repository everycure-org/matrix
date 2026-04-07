-- All minor components (non-LCC) with size >= 2, plus any with core entities
SELECT
  CAST(component_id AS INT) as component_id,
  component_size,
  num_drugs,
  num_diseases,
  num_other
FROM `${project_id}.release_${bq_release_version}.connected_components`
WHERE component_id > 0
  AND (component_size >= 2 OR num_drugs > 0 OR num_diseases > 0)
ORDER BY component_size DESC
