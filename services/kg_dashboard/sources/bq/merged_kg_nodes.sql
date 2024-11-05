SELECT
    category, 
    SPLIT(id, ':')[OFFSET(0)] as prefix,
    ARRAY_TO_STRING(upstream_data_source, ', ') AS upstream_data_source,
    count(*) as count
FROM  `mtrx-hub-dev-3of.release_v0_2_2.nodes`
group by all
