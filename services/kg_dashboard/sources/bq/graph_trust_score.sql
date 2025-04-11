WITH mapped AS (
  SELECT
    knowledge_level,
    CASE
      WHEN knowledge_level = 'knowledge_assertion' THEN 1.0
      WHEN knowledge_level = 'logical_entailment' THEN 0.99
      WHEN knowledge_level = 'prediction' THEN 0.5
      WHEN knowledge_level = 'statistical_association' THEN 0.5
      WHEN knowledge_level = 'observation' THEN 0.5
      WHEN knowledge_level = 'not_provided' THEN 0.25
    END AS trust_score
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.edges`
  WHERE knowledge_level IS NOT NULL AND knowledge_level != ''
)

SELECT
  COUNT(*) AS included_edges,
  ROUND(AVG(trust_score), 4) AS overall_graph_trust_score
FROM mapped
WHERE trust_score IS NOT NULL;