WITH mapped AS (
  SELECT
    knowledge_level,
    (2 * CASE
      WHEN knowledge_level = 'knowledge_assertion' THEN 1.0
      WHEN knowledge_level = 'logical_entailment' THEN 0.85
      WHEN knowledge_level = 'statistical_association' THEN 0.65
      WHEN knowledge_level = 'observation' THEN 0.4
      WHEN knowledge_level = 'prediction' THEN 0.2
      WHEN knowledge_level = 'not_provided' THEN 0.5
      WHEN knowledge_level IS NULL OR knowledge_level = '' THEN 0.5
      ELSE 0.5
    END) - 1 AS knowledge_level_score
  FROM `${project_id}.release_${bq_release_version}.edges_unified`
)

SELECT
  COUNT(*) AS included_edges,
  ROUND(AVG(knowledge_level_score), 4) AS average_knowledge_level
FROM mapped
WHERE knowledge_level_score IS NOT NULL;