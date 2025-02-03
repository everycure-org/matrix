SELECT
    category, 
    SPLIT(id, ':')[OFFSET(0)] as prefix,
    upstream_data_source.list AS upstream_data_source,
    count(*) as count
FROM  `mtrx-hub-dev-3of.release_${release_version}.nodes`
group by all
