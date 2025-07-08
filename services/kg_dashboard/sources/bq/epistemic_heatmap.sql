WITH base AS (
  SELECT
    knowledge_level,
    agent_type,
    CASE
      WHEN knowledge_level = 'knowledge_assertion' THEN 'Knowledge Assertion'
      WHEN knowledge_level = 'logical_entailment' THEN 'Logical Entailment'
      WHEN knowledge_level = 'prediction' THEN 'Prediction'
      WHEN knowledge_level = 'statistical_association' THEN 'Statistical Association'
      WHEN knowledge_level = 'observation' THEN 'Observation'
      WHEN knowledge_level = 'not_provided' OR knowledge_level IS NULL OR knowledge_level = '' THEN 'Not Provided'
      ELSE 'Other'
    END AS knowledge_level_label,

    CASE
      WHEN agent_type = 'manual_agent' THEN 'Manual Agent'
      WHEN agent_type = 'manual_validation_of_automated_agent' THEN 'Manual Validation of Automated Agent'
      WHEN agent_type = 'automated_agent' THEN 'Automated Agent'
      WHEN agent_type = 'data_analysis_pipeline' THEN 'Data Analysis Pipeline'
      WHEN agent_type = 'computational_model' THEN 'Computational Model'
      WHEN agent_type = 'text_mining_agent' THEN 'Text Mining Agent'
      WHEN agent_type = 'image_processing_agent' THEN 'Image Processing Agent'
      WHEN agent_type = 'not_provided' OR agent_type IS NULL OR agent_type = '' THEN 'Not Provided'
      ELSE 'Other'
    END AS agent_type_label
  FROM `${project_id}.release_${bq_release_version}.edges_unified`
),

scored AS (
  SELECT
    knowledge_level_label,
    agent_type_label,
    (2 * CASE knowledge_level
      WHEN 'knowledge_assertion' THEN 1.0
      WHEN 'logical_entailment' THEN 0.85
      WHEN 'statistical_association' THEN 0.65
      WHEN 'observation' THEN 0.4
      WHEN 'prediction' THEN 0.2
      WHEN 'not_provided' THEN 0.5
      WHEN NULL THEN 0.5
      ELSE 0.5
    END) - 1 AS kl_score,

    (2 * CASE agent_type
      WHEN 'manual_agent' THEN 1.0
      WHEN 'manual_validation_of_automated_agent' THEN 0.95
      WHEN 'data_analysis_pipeline' THEN 0.7
      WHEN 'automated_agent' THEN 0.6
      WHEN 'computational_model' THEN 0.5
      WHEN 'text_mining_agent' THEN 0.35
      WHEN 'image_processing_agent' THEN 0.35
      WHEN 'not_provided' THEN 0.5
      WHEN NULL THEN 0.5
      ELSE 0.5
    END) - 1 AS at_score,
  FROM base
)

SELECT
  knowledge_level_label,
  agent_type_label,
  COUNT(*) AS edge_count,
  ROUND(AVG((kl_score + at_score) / 2), 3) AS average_score
FROM scored
GROUP BY knowledge_level_label, agent_type_label