SELECT
  CASE
    WHEN knowledge_level = 'knowledge_assertion' THEN 'Knowledge Assertion'
    WHEN knowledge_level = 'logical_entailment' THEN 'Logical Entailment'
    WHEN knowledge_level = 'prediction' THEN 'Prediction'
    WHEN knowledge_level = 'statistical_association' THEN 'Statistical Association'
    WHEN knowledge_level = 'observation' THEN 'Observation'
    WHEN knowledge_level = 'not_provided' THEN 'Not Provided'
    ELSE 'Other'
  END AS knowledge_level_label,
  CASE
    WHEN agent_type = 'manual_agent' THEN 'Manual Agent'
    WHEN agent_type = 'manual_validation_of_automated_agent' THEN 'Manual Validation of Automated Agent'
    WHEN agent_type = 'automated_agent' THEN 'Automated Agent'
    WHEN agent_type = 'data_analysis_pipeline' THEN 'Data Analysis Pipeline'
    WHEN agent_type = 'computational_model' THEN 'Computational Model'
    WHEN agent_type = 'text_mining_agent' THEN 'Text Mining Agent'
    WHEN agent_type = 'not_provided' THEN 'Not Provided'
    ELSE 'Other'
  END AS agent_type_label,
  COUNT(*) AS edge_count,
  ROUND(AVG((
    CASE knowledge_level
      WHEN 'knowledge_assertion' THEN 1.0
      WHEN 'logical_entailment' THEN 0.99
      WHEN 'prediction' THEN 0.5
      WHEN 'statistical_association' THEN 0.5
      WHEN 'observation' THEN 0.5
      WHEN 'not_provided' THEN 0.25
      ELSE 0.0
    END +
    CASE agent_type
      WHEN 'manual_agent' THEN 1.0
      WHEN 'manual_validation_of_automated_agent' THEN 1.0
      WHEN 'automated_agent' THEN 0.75
      WHEN 'data_analysis_pipeline' THEN 0.5
      WHEN 'computational_model' THEN 0.5
      WHEN 'text_mining_agent' THEN 0.3
      WHEN 'not_provided' THEN 0.25
      ELSE 0.0
    END
  ) / 2), 3) AS average_score
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.edges`
WHERE knowledge_level IS NOT NULL AND agent_type IS NOT NULL
GROUP BY knowledge_level_label, agent_type_label