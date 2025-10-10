-- Pre-compute connected categories for all key nodes (for Sankey diagrams)

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

-- Incoming connections: Subject Categories to Key Node descendants
SELECT
  descendants.key_node_id,
  CONCAT('[IN] ', REPLACE(subject_nodes.category, 'biolink:', '')) as source,
  'Key Node' as target,
  COUNT(*) as count
FROM descendants
JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
  ON edges.object = descendants.descendant_id
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` subject_nodes
  ON edges.subject = subject_nodes.id
GROUP BY descendants.key_node_id, subject_nodes.category
HAVING COUNT(*) > 100

UNION ALL

-- Outgoing connections: Key Node descendants to Object Categories
SELECT
  descendants.key_node_id,
  'Key Node' as source,
  CONCAT('[OUT] ', REPLACE(object_nodes.category, 'biolink:', '')) as target,
  COUNT(*) as count
FROM descendants
JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
  ON edges.subject = descendants.descendant_id
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` object_nodes
  ON edges.object = object_nodes.id
GROUP BY descendants.key_node_id, object_nodes.category
HAVING COUNT(*) > 100

ORDER BY key_node_id, count DESC
