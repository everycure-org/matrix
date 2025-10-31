-- Key nodes list with basic statistics (no recursion)
-- Shows all configured key disease and drug nodes

WITH key_node_ids AS (
  SELECT id FROM UNNEST(SPLIT('${key_disease_ids}', ',')) AS id
  UNION ALL
  SELECT id FROM UNNEST(SPLIT('${key_drug_ids}', ',')) AS id
),

node_info AS (
  SELECT
    n.id,
    n.name,
    n.category
  FROM key_node_ids k
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` n
    ON n.id = k.id
),

edge_counts AS (
  SELECT
    node_info.id,
    COUNT(DISTINCT CASE WHEN edges.subject = node_info.id THEN edges.subject || edges.predicate || edges.object END) as outgoing_edges,
    COUNT(DISTINCT CASE WHEN edges.object = node_info.id THEN edges.subject || edges.predicate || edges.object END) as incoming_edges,
    COUNT(DISTINCT CASE WHEN edges.subject = node_info.id THEN edges.object END) as unique_outgoing_neighbors,
    COUNT(DISTINCT CASE WHEN edges.object = node_info.id THEN edges.subject END) as unique_incoming_neighbors
  FROM node_info
  LEFT JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.subject = node_info.id OR edges.object = node_info.id
  GROUP BY node_info.id
)

SELECT
  node_info.id,
  node_info.name,
  node_info.category,
  COALESCE(edge_counts.outgoing_edges, 0) + COALESCE(edge_counts.incoming_edges, 0) as total_edges,
  COALESCE(edge_counts.unique_outgoing_neighbors, 0) + COALESCE(edge_counts.unique_incoming_neighbors, 0) as unique_neighbors,
  '/Key Nodes/' || node_info.id as link
FROM node_info
LEFT JOIN edge_counts ON edge_counts.id = node_info.id
ORDER BY node_info.category, node_info.name
