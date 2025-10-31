WITH scored_edges AS (
  SELECT
    primary_knowledge_source,
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
),

most_common_values AS (
  SELECT
    primary_knowledge_source,
    -- Get most common knowledge_level (handle nulls as 'not_provided')
    ARRAY_AGG(
      COALESCE(NULLIF(knowledge_level, ''), 'not_provided')
      ORDER BY knowledge_level_count DESC
      LIMIT 1
    )[OFFSET(0)] AS most_common_knowledge_level,
    -- Get most common agent_type (handle nulls as 'not_provided')
    ARRAY_AGG(
      COALESCE(NULLIF(agent_type, ''), 'not_provided')
      ORDER BY agent_type_count DESC
      LIMIT 1
    )[OFFSET(0)] AS most_common_agent_type
  FROM (
    SELECT
      primary_knowledge_source,
      COALESCE(NULLIF(knowledge_level, ''), 'not_provided') AS knowledge_level,
      COALESCE(NULLIF(agent_type, ''), 'not_provided') AS agent_type,
      COUNT(*) OVER (PARTITION BY primary_knowledge_source, knowledge_level) AS knowledge_level_count,
      COUNT(*) OVER (PARTITION BY primary_knowledge_source, agent_type) AS agent_type_count
    FROM `${project_id}.release_${bq_release_version}.edges_unified`
  )
  GROUP BY primary_knowledge_source
)

SELECT
  s.primary_knowledge_source,
  COUNT(*) AS included_edges,
  ROUND(AVG((s.kl_score + s.at_score) / 2), 4) AS average_epistemic_score,
  ROUND(AVG(s.kl_score), 4) AS average_knowledge_level_score,
  ROUND(AVG(s.at_score), 4) AS average_agent_type_score,
  COUNTIF(
    (s.knowledge_level IS NULL OR s.knowledge_level = 'not_provided' OR s.knowledge_level = '') AND
    (s.agent_type IS NULL OR s.agent_type = 'not_provided' OR s.agent_type = '')
  ) AS null_or_not_provided_both,
  m.most_common_knowledge_level,
  m.most_common_agent_type
FROM scored_edges s
LEFT JOIN most_common_values m ON s.primary_knowledge_source = m.primary_knowledge_source
GROUP BY s.primary_knowledge_source, m.most_common_knowledge_level, m.most_common_agent_type
ORDER BY average_epistemic_score DESC;
