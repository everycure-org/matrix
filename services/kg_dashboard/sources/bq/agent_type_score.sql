WITH mapped AS (
  SELECT
    agent_type,
    (2 * CASE
      WHEN agent_type = 'manual_agent' THEN 1.0
      WHEN agent_type = 'manual_validation_of_automated_agent' THEN 0.95
      WHEN agent_type = 'data_analysis_pipeline' THEN 0.7
      WHEN agent_type = 'automated_agent' THEN 0.6
      WHEN agent_type = 'computational_model' THEN 0.5
      WHEN agent_type = 'text_mining_agent' THEN 0.35
      WHEN agent_type = 'image_processing_agent' THEN 0.35
      WHEN agent_type = 'not_provided' THEN 0.5
      WHEN agent_type IS NULL OR agent_type = '' THEN 0.5
      ELSE 0.5
    END) - 1 AS agent_type_score
  FROM `${project_id}.release_${bq_release_version}.edges_unified`
)

SELECT
  COUNT(*) AS included_edges,
  ROUND(AVG(agent_type_score), 4) AS average_agent_type
FROM mapped
WHERE agent_type_score IS NOT NULL;