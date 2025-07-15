WITH prepared AS (
  SELECT
    CASE
      WHEN LOWER(TRIM(agent_type)) = 'manual_agent' THEN 'Manual Agent'
      WHEN LOWER(TRIM(agent_type)) = 'manual_validation_of_automated_agent' THEN 'Manual Validation of Automated Agent'
      WHEN LOWER(TRIM(agent_type)) = 'automated_agent' THEN 'Automated Agent'
      WHEN LOWER(TRIM(agent_type)) = 'data_analysis_pipeline' THEN 'Data Analysis Pipeline'
      WHEN LOWER(TRIM(agent_type)) = 'computational_model' THEN 'Computational Model'
      WHEN LOWER(TRIM(agent_type)) = 'text_mining_agent' THEN 'Text-Mining Agent'
      WHEN LOWER(TRIM(agent_type)) = 'image_processing_agent' THEN 'Image-Processing Agent'
      WHEN LOWER(TRIM(agent_type)) = 'not_provided' THEN 'Not Provided'
      WHEN agent_type IS NULL OR agent_type = '' THEN 'null'
      ELSE 'null'
    END AS agent_type,
    src_element.element AS upstream_data_source
  FROM `${project_id}.release_${bq_release_version}.edges_unified`,
    UNNEST(upstream_data_source.list) AS src_element
)

SELECT
  agent_type,
  upstream_data_source,
  COUNT(*) AS edge_count
FROM prepared
GROUP BY agent_type, upstream_data_source
HAVING COUNT(*) > 0