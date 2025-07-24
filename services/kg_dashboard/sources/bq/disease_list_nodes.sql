SELECT 
    disease_list_nodes.id,
    disease_list_nodes.name,
    COALESCE(COUNT(*), 0) as edge_count 
FROM `${project_id}.release_${bq_release_version}.disease_list_nodes_normalized` as disease_list_nodes
LEFT OUTER JOIN `${project_id}.release_${bq_release_version}.edges_unified` as edges
    ON disease_list_nodes.id = edges.subject 
    OR disease_list_nodes.id = edges.object
GROUP BY ALL