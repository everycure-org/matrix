WITH base_edges AS (
  SELECT
    primary_knowledge_source,
    COALESCE(NULLIF(TRIM(knowledge_level), ''), 'not_provided') AS kl_normalized,
    COALESCE(NULLIF(TRIM(agent_type), ''), 'not_provided') AS at_normalized,
    src_element.element AS upstream_data_source
  FROM `${project_id}.release_${bq_release_version}.edges_unified`,
    UNNEST(upstream_data_source.list) AS src_element
),

scored_edges AS (
  SELECT
    primary_knowledge_source,
    kl_normalized,
    at_normalized,
    upstream_data_source,

    -- Knowledge Level Score: -1 (weakest) to 1 (strongest)
    (2 * CASE LOWER(kl_normalized)
      WHEN 'knowledge_assertion' THEN 1.0
      WHEN 'logical_entailment' THEN 0.85
      WHEN 'statistical_association' THEN 0.65
      WHEN 'observation' THEN 0.4
      WHEN 'prediction' THEN 0.2
      WHEN 'not_provided' THEN 0.5
      ELSE 0.5
    END) - 1 AS kl_score,

    -- Agent Type Score: -1 (weakest) to 1 (strongest)
    (2 * CASE LOWER(at_normalized)
      WHEN 'manual_agent' THEN 1.0
      WHEN 'manual_validation_of_automated_agent' THEN 0.95
      WHEN 'data_analysis_pipeline' THEN 0.7
      WHEN 'automated_agent' THEN 0.6
      WHEN 'computational_model' THEN 0.5
      WHEN 'text_mining_agent' THEN 0.35
      WHEN 'image_processing_agent' THEN 0.35
      WHEN 'not_provided' THEN 0.5
      ELSE 0.5
    END) - 1 AS at_score,

    -- Human-readable labels
    CASE LOWER(kl_normalized)
      WHEN 'knowledge_assertion' THEN 'Knowledge Assertion'
      WHEN 'logical_entailment' THEN 'Logical Entailment'
      WHEN 'prediction' THEN 'Prediction'
      WHEN 'statistical_association' THEN 'Statistical Association'
      WHEN 'observation' THEN 'Observation'
      WHEN 'not_provided' THEN 'Not Provided'
      ELSE 'Not Provided'
    END AS knowledge_level_label,

    CASE LOWER(at_normalized)
      WHEN 'manual_agent' THEN 'Manual Agent'
      WHEN 'manual_validation_of_automated_agent' THEN 'Manual Validation\nof Automated Agent'
      WHEN 'automated_agent' THEN 'Automated Agent'
      WHEN 'data_analysis_pipeline' THEN 'Data Analysis Pipeline'
      WHEN 'computational_model' THEN 'Computational Model'
      WHEN 'text_mining_agent' THEN 'Text-Mining Agent'
      WHEN 'image_processing_agent' THEN 'Image Processing\nAgent'
      WHEN 'not_provided' THEN 'Not Provided'
      ELSE 'Not Provided'
    END AS agent_type_label
  FROM base_edges
)

SELECT
  primary_knowledge_source,
  kl_normalized,
  at_normalized,
  kl_score,
  at_score,
  knowledge_level_label,
  agent_type_label,
  upstream_data_source,
  COUNT(*) AS edge_count
FROM scored_edges
GROUP BY
  primary_knowledge_source,
  kl_normalized,
  at_normalized,
  kl_score,
  at_score,
  knowledge_level_label,
  agent_type_label,
  upstream_data_source
