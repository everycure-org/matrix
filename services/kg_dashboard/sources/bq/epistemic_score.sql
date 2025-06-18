WITH scored_edges AS (
  SELECT
    knowledge_level,
    agent_type,
    (2 * CASE
      WHEN knowledge_level = 'knowledge_assertion' THEN 1.0
      WHEN knowledge_level = 'logical_entailment' THEN 0.85
      WHEN knowledge_level = 'statistical_association' THEN 0.65
      WHEN knowledge_level = 'observation' THEN 0.4
      WHEN knowledge_level = 'prediction' THEN 0.2
      WHEN knowledge_level = 'not_provided' THEN 0.5
      WHEN knowledge_level IS NULL OR knowledge_level = '' THEN 0.5
      ELSE 0.5
    END) - 1 AS kl_score,
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
    END) - 1 AS at_score
  FROM `${project_id}.release_${bq_release_version}.edges_unified`
)

SELECT
  COUNT(*) AS included_edges,
  ROUND(AVG((kl_score + at_score) / 2), 4) AS average_epistemic_score,
  COUNTIF(
    (knowledge_level IS NULL OR knowledge_level = 'not_provided' OR knowledge_level = '') AND
    (agent_type IS NULL OR agent_type = 'not_provided' OR agent_type = '')
  ) AS null_or_not_provided_both
FROM scored_edges;
