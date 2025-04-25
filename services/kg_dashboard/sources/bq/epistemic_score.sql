WITH scored_edges AS (
  SELECT
    knowledge_level,
    agent_type,
    CASE
      WHEN knowledge_level = 'knowledge_assertion' THEN 1.0
      WHEN knowledge_level = 'logical_entailment' THEN 0.99
      WHEN knowledge_level = 'prediction' THEN 0.5
      WHEN knowledge_level = 'statistical_association' THEN 0.5
      WHEN knowledge_level = 'observation' THEN 0.5
      WHEN knowledge_level = 'not_provided' THEN 0.25
    END AS kl_score,
    CASE
      WHEN agent_type = 'manual_agent' THEN 1.0
      WHEN agent_type = 'manual_validation_of_automated_agent' THEN 1.0
      WHEN agent_type = 'automated_agent' THEN 0.75
      WHEN agent_type = 'data_analysis_pipeline' THEN 0.5
      WHEN agent_type = 'computational_model' THEN 0.5
      WHEN agent_type = 'text_mining_agent' THEN 0.3
      WHEN agent_type = 'image_processing_agent' THEN 0.3
      WHEN agent_type = 'not_provided' THEN 0.25
    END AS at_score
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.edges`
  WHERE
    knowledge_level IS NOT NULL AND knowledge_level != ''
    AND agent_type IS NOT NULL AND agent_type != ''
)

SELECT
    COUNT(*) AS included_edges,
    ROUND(AVG((kl_score + at_score) / 2), 4) AS average_epistemic_score
FROM scored_edges
