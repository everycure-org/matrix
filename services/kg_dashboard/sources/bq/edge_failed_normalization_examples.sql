WITH failed_normalizations AS (
    SELECT DISTINCT
        u.id,
        u.original_id,
        SPLIT(u.id, ':')[OFFSET(0)] AS prefix,
        u.all_categories.list[SAFE_OFFSET(0)].element AS category,
        u.upstream_data_source AS normalization_set
    FROM
        `${project_id}.release_${bq_release_version}.unified_normalization_summary` u
    WHERE
        u.normalization_success = false
        AND u.id != "['Error']"
),
with_names AS (
    SELECT
        f.id,
        COALESCE(n.name, f.id) AS name,
        f.prefix,
        f.category,
        f.normalization_set
    FROM
        failed_normalizations f
    LEFT JOIN
        `${project_id}.release_${bq_release_version}.nodes_unified` n
        ON f.id = n.id
),
ranked_data AS (
    SELECT
        id,
        name,
        prefix,
        category,
        normalization_set,
        ROW_NUMBER() OVER (PARTITION BY normalization_set, prefix ORDER BY id) AS row_num
    FROM
        with_names
)
SELECT
    id,
    name,
    prefix,
    category,
    normalization_set
FROM
    ranked_data
WHERE
    row_num <= ${max_edge_failed_normalization_by_normalization_set_prefix}
