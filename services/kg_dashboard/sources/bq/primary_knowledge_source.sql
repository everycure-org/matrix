SELECT
  primary_knowledge_source AS source,
  aggregator.element AS target,
  COUNT(*) * 1.0 AS count
FROM
  `mtrx-hub-dev-3of.release_${bq_release_version}.edges_unified`,
  UNNEST(aggregator_knowledge_source.list) AS aggregator
GROUP BY
  primary_knowledge_source, aggregator.element
ORDER BY
  count DESC
