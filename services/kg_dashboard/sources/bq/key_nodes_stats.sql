-- Pre-compute statistics for each key node including descendants

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
    AND edges.primary_knowledge_source in ('infores:mondo', 'infores:chebi')
  WHERE descendants.depth < 20
),

node_info AS (
  SELECT DISTINCT
    descendants.key_node_id,
    nodes.name,
    nodes.category
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` nodes
    ON nodes.id = descendants.key_node_id
),

-- Direct edges (key node only, no descendants)
direct_edges AS (
  SELECT
    node_info.key_node_id,
    COUNT(DISTINCT CASE WHEN edges.subject = node_info.key_node_id THEN edges.subject || '|' || edges.predicate || '|' || edges.object END) as outgoing_edges,
    COUNT(DISTINCT CASE WHEN edges.object = node_info.key_node_id THEN edges.subject || '|' || edges.predicate || '|' || edges.object END) as incoming_edges,
    COUNT(DISTINCT CASE WHEN edges.subject = node_info.key_node_id THEN edges.object END) as unique_outgoing_neighbors,
    COUNT(DISTINCT CASE WHEN edges.object = node_info.key_node_id THEN edges.subject END) as unique_incoming_neighbors
  FROM node_info
  LEFT JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.subject = node_info.key_node_id OR edges.object = node_info.key_node_id
  GROUP BY node_info.key_node_id
),

-- Descendant statistics
descendant_stats AS (
  SELECT
    descendants.key_node_id,
    COUNT(DISTINCT descendants.descendant_id) as descendant_count,
    MAX(descendants.depth) as max_depth
  FROM descendants
  GROUP BY descendants.key_node_id
),

-- Edges including all descendants
descendant_edges AS (
  SELECT
    descendants.key_node_id,
    COUNT(DISTINCT CASE WHEN edges.subject = descendants.descendant_id THEN edges.subject || '|' || edges.predicate || '|' || edges.object END) as outgoing_edges,
    COUNT(DISTINCT CASE WHEN edges.object = descendants.descendant_id THEN edges.subject || '|' || edges.predicate || '|' || edges.object END) as incoming_edges,
    COUNT(DISTINCT CASE WHEN edges.subject = descendants.descendant_id THEN edges.object END) as unique_outgoing_neighbors,
    COUNT(DISTINCT CASE WHEN edges.object = descendants.descendant_id THEN edges.subject END) as unique_incoming_neighbors
  FROM descendants
  LEFT JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.subject = descendants.descendant_id OR edges.object = descendants.descendant_id
  GROUP BY descendants.key_node_id
)

SELECT
  node_info.key_node_id as id,
  node_info.name,
  node_info.category,
  descendant_stats.descendant_count,
  descendant_stats.max_depth as max_descendant_depth,
  direct_edges.outgoing_edges + direct_edges.incoming_edges as direct_total_edges,
  direct_edges.unique_outgoing_neighbors + direct_edges.unique_incoming_neighbors as direct_unique_neighbors,
  descendant_edges.outgoing_edges + descendant_edges.incoming_edges as with_descendants_total_edges,
  descendant_edges.unique_outgoing_neighbors + descendant_edges.unique_incoming_neighbors as with_descendants_unique_neighbors
FROM node_info
LEFT JOIN direct_edges ON direct_edges.key_node_id = node_info.key_node_id
LEFT JOIN descendant_stats ON descendant_stats.key_node_id = node_info.key_node_id
LEFT JOIN descendant_edges ON descendant_edges.key_node_id = node_info.key_node_id
