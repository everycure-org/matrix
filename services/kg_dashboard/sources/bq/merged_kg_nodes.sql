-- TODO: rename source file to include groupby/count by category information
SELECT
    category, 
    SPLIT(id, ':')[OFFSET(0)] AS prefix,
    STRING_AGG(DISTINCT upstream_data_source.list[SAFE_OFFSET(0)].element, ',' ORDER BY upstream_data_source.list[SAFE_OFFSET(0)].element) AS upstream_data_source,
    '${bq_release_version}' AS release_version,
    count(*) as count
FROM  `${project_id}.release_${bq_release_version}.nodes_unified`
GROUP BY all
