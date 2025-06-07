SELECT 
  STRING_AGG(DISTINCT edges.upstream_data_source.list[SAFE_OFFSET(0)].element, ',' ORDER BY edges.upstream_data_source.list[SAFE_OFFSET(0)].element) AS upstream_data_source,  
  primary_knowledge_source,
  replace(subject_nodes.category, 'biolink:','') 
    || '-' || replace(predicate, 'biolink:', '') 
    || '-' || replace(object_nodes.category, 'biolink:','') as edge_type,
  (2 * ROUND(AVG(CASE
      WHEN knowledge_level = 'knowledge_assertion' THEN 1.0
      WHEN knowledge_level = 'logical_entailment' THEN 0.85
      WHEN knowledge_level = 'statistical_association' THEN 0.65
      WHEN knowledge_level = 'observation' THEN 0.4
      WHEN knowledge_level = 'prediction' THEN 0.2
      WHEN knowledge_level = 'not_provided' THEN 0.5
      WHEN knowledge_level IS NULL OR knowledge_level = '' THEN 0.5
      ELSE 0.5
    END),4)) - 1 AS knowledge_level_score,
  count(*) as count
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.edges_unified` AS edges
  JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_unified` AS subject_nodes ON edges.subject = subject_nodes.id
  JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_unified` AS object_nodes ON edges.object = object_nodes.id
GROUP BY ALL