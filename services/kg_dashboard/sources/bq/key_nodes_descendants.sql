-- Compute all descendants for each key node using recursive CTE

WITH RECURSIVE
key_node_ids AS (
  SELECT id FROM UNNEST(SPLIT('${key_disease_ids}', ',')) AS id
  UNION ALL
  SELECT id FROM UNNEST(SPLIT('${key_drug_ids}', ',')) AS id
),

descendants AS (
  -- Base case: each key node itself at depth 0
  SELECT
    key_node_ids.id as key_node_id,
    key_node_ids.id as descendant_id,
    0 as depth
  FROM key_node_ids

  UNION ALL

  -- Recursive case: follow subclass_of edges
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
  key_node_id,
  descendant_id,
  depth
FROM descendants
ORDER BY key_node_id, depth, descendant_id
