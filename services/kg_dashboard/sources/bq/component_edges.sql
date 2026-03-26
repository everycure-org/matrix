-- Edges for the 50 largest minor components plus any with core entities
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
),
component_node_ids AS (
  SELECT id, component_id
  FROM `${project_id}.release_${bq_release_version}.node_metrics`
  WHERE component_id IN (SELECT component_id FROM target_components)
)
SELECT
  s.component_id,
  e.subject,
  sn.name as subject_name,
  REPLACE(sn.category, 'biolink:', '') as subject_category,
  REPLACE(e.predicate, 'biolink:', '') as predicate,
  e.object,
  on_.name as object_name,
  REPLACE(on_.category, 'biolink:', '') as object_category,
  REPLACE(e.primary_knowledge_source, 'infores:', '') as primary_knowledge_source
FROM `${project_id}.release_${bq_release_version}.edges_unified` e
JOIN component_node_ids s ON e.subject = s.id
JOIN component_node_ids o ON e.object = o.id AND s.component_id = o.component_id
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` sn ON e.subject = sn.id
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` on_ ON e.object = on_.id
ORDER BY s.component_id, subject_category, predicate, object_category
