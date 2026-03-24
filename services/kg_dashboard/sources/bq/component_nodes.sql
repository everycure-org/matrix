-- Nodes for the 50 largest minor components plus any with core entities
-- Used by the component detail parameterized page
WITH top_by_size AS (
  SELECT component_id
  FROM `${project_id}.release_${bq_release_version}.connected_components`
  WHERE component_id > 0
  ORDER BY component_size DESC
  LIMIT 50
),
with_core AS (
  SELECT component_id
  FROM `${project_id}.release_${bq_release_version}.connected_components`
  WHERE component_id > 0
    AND (num_drugs > 0 OR num_diseases > 0)
),
target_components AS (
  SELECT component_id FROM top_by_size
  UNION DISTINCT
  SELECT component_id FROM with_core
)
SELECT
  nm.component_id,
  nm.id,
  n.name,
  REPLACE(n.category, 'biolink:', '') as category,
  nm.ec_core_category
FROM `${project_id}.release_${bq_release_version}.node_metrics` nm
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` n ON nm.id = n.id
WHERE nm.component_id IN (SELECT component_id FROM target_components)
ORDER BY nm.component_id, n.category, n.name
