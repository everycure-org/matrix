WITH mapped AS (
  SELECT
    agent_type,
    CASE
      WHEN agent_type = 'manual_agent' THEN 1.0
      WHEN agent_type = 'manual_validation_of_automated_agent' THEN 1.0
      WHEN agent_type = 'automated_agent' THEN 0.75
      WHEN agent_type = 'data_analysis_pipeline' THEN 0.5
      WHEN agent_type = 'computational_model' THEN 0.5
      WHEN agent_type = 'text_mining_agent' THEN 0.3
      WHEN agent_type = 'not_provided' THEN 0.25
    END AS agent_type_score
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.edges`
  WHERE agent_type IS NOT NULL AND agent_type != ''
)

SELECT
  COUNT(*) AS included_edges,
  ROUND(AVG(agent_type_score), 4) AS average_agent_type
FROM mapped
WHERE agent_type_score IS NOT NULL;