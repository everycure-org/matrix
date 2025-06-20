WITH prepared AS (
  SELECT
    CASE
      WHEN LOWER(TRIM(knowledge_level)) = 'knowledge_assertion' THEN 'Knowledge Assertion'
      WHEN LOWER(TRIM(knowledge_level)) = 'logical_entailment' THEN 'Logical Entailment'
      WHEN LOWER(TRIM(knowledge_level)) = 'prediction' THEN 'Prediction'
      WHEN LOWER(TRIM(knowledge_level)) = 'statistical_association' THEN 'Statistical Association'
      WHEN LOWER(TRIM(knowledge_level)) = 'observation' THEN 'Observation'
      WHEN LOWER(TRIM(knowledge_level)) = 'not_provided' THEN 'Not Provided'
      WHEN knowledge_level IS NULL OR knowledge_level = '' THEN 'null'
      ELSE 'null'
    END AS knowledge_level,
    src_element.element AS upstream_data_source
  FROM `${project_id}.release_${bq_release_version}.edges_unified`,
    UNNEST(upstream_data_source.list) AS src_element
)

SELECT
  knowledge_level,
  upstream_data_source,
  COUNT(*) AS edge_count
FROM prepared
GROUP BY knowledge_level, upstream_data_source
HAVING COUNT(*) > 0
