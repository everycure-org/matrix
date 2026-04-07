-- Knowledge source composition of all minor (non-LCC) components
SELECT
  REPLACE(e.primary_knowledge_source, 'infores:', '') as knowledge_source,
  COUNT(*) as edge_count,
  COUNT(DISTINCT s.component_id) as component_count,
  STRING_AGG(DISTINCT CAST(CAST(s.component_id AS INT) AS STRING), ', ' ORDER BY CAST(CAST(s.component_id AS INT) AS STRING)) as component_ids
FROM `${project_id}.release_${bq_release_version}.edges_unified` e
JOIN `${project_id}.release_${bq_release_version}.node_metrics` s ON e.subject = s.id
JOIN `${project_id}.release_${bq_release_version}.node_metrics` o ON e.object = o.id
  AND s.component_id = o.component_id
WHERE s.component_id > 0
GROUP BY knowledge_source
ORDER BY edge_count DESC
