SELECT
    category, 
    SPLIT(id, ':')[OFFSET(0)] as prefix,
    STRING_AGG(DISTINCT upstream_data_source.list[SAFE_OFFSET(0)].element, ',' ORDER BY upstream_data_source.list[SAFE_OFFSET(0)].element) AS upstream_data_source,
    '${release_version}' AS release_version,
    count(*) as count
FROM  `mtrx-hub-dev-3of.release_${release_version}.nodes`
group by all
