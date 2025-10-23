-- Pre-compute edge breakdown for all key nodes with descendants

WITH RECURSIVE
key_node_ids AS (
  SELECT id FROM UNNEST(SPLIT('${key_disease_ids}', ',')) AS id
  UNION ALL
  SELECT id FROM UNNEST(SPLIT('${key_drug_ids}', ',')) AS id
),

descendants AS (
  SELECT
    key_node_ids.id as key_node_id,
    key_node_ids.id as descendant_id,
    0 as depth
  FROM key_node_ids

  UNION ALL

  SELECT
    descendants.key_node_id,
    edges.subject as descendant_id,
    descendants.depth + 1 as depth
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.object = descendants.descendant_id
    AND edges.predicate = 'biolink:subclass_of'
  WHERE descendants.depth < 20
    AND EXISTS(
      SELECT 1 FROM UNNEST(edges.primary_knowledge_sources.list) AS pks
      WHERE pks.element IN ('infores:mondo', 'infores:chebi')
    )
)

SELECT
  descendants.key_node_id,
  edges.predicate,
  subject_nodes.category as subject_category,
  object_nodes.category as object_category,
  COUNT(*) as edge_count,
  COUNT(DISTINCT edges.subject) as unique_subjects,
  COUNT(DISTINCT edges.object) as unique_objects
FROM descendants
JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
  ON edges.subject = descendants.descendant_id OR edges.object = descendants.descendant_id
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` subject_nodes
  ON edges.subject = subject_nodes.id
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` object_nodes
  ON edges.object = object_nodes.id
GROUP BY descendants.key_node_id, edges.predicate, subject_nodes.category, object_nodes.category
ORDER BY descendants.key_node_id, edge_count DESC
