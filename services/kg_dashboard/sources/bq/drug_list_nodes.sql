SELECT 
    drug_list_nodes.id,
    drug_list_nodes.name,
    COALESCE(COUNT(*), 0) as edge_count 
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` as drug_list_nodes
LEFT OUTER JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.edges_unified` as edges
    ON drug_list_nodes.id = edges.subject 
    OR drug_list_nodes.id = edges.object
GROUP BY ALL