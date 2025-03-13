WITH ranked_data AS (
    SELECT 
        id,
        name,
        prefix,
        category,
        normalization_set,
        ROW_NUMBER() OVER (PARTITION BY prefix ORDER BY id) AS row_num
    FROM (
        SELECT 
            subject AS id,
            name,
            SPLIT(subject, ':')[OFFSET(0)] AS prefix,
            category,
            'rtx_kg2' AS normalization_set,
        FROM 
            `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_edges_normalized`
            JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_nodes_normalized` ON subject = id
        WHERE subject_normalization_success = false
        UNION DISTINCT
        SELECT 
            object AS id,
            name,
            SPLIT(object, ':')[OFFSET(0)] AS prefix, 
            category,
            'rtx_kg2' AS normalization_set,
        FROM 
            `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_edges_normalized`
            JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_nodes_normalized` ON object = id
        WHERE object_normalization_success = false        
        UNION DISTINCT
        SELECT 
            subject AS id,
            name,
            SPLIT(subject, ':')[OFFSET(0)] AS prefix,
            category,
            'robokop' AS normalization_set,
        FROM 
            `mtrx-hub-dev-3of.release_${bq_release_version}.robokop_edges_normalized`
            JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.robokop_nodes_normalized` ON subject = id
        WHERE subject_normalization_success = false
        UNION DISTINCT
        SELECT 
            object AS id,
            name,
            SPLIT(object, ':')[OFFSET(0)] AS prefix, 
            category,
            'robokop' AS normalization_set,
        FROM 
            `mtrx-hub-dev-3of.release_${bq_release_version}.robokop_edges_normalized`
            JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.robokop_nodes_normalized` ON object = id
        WHERE object_normalization_success = false
        UNION DISTINCT
        SELECT 
            subject AS id,
            name,
            SPLIT(subject, ':')[OFFSET(0)] AS prefix,
            category,
            'ground_truth' AS normalization_set,
        FROM 
            `mtrx-hub-dev-3of.release_${bq_release_version}.ground_truth_edges_normalized`        
            JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_nodes_normalized` ON subject = id
        WHERE subject_normalization_success = false
        UNION DISTINCT
        SELECT 
            object AS id,
            name,
            SPLIT(object, ':')[OFFSET(0)] AS prefix, 
            category,
            'ground_truth' AS normalization_set,
        FROM 
            `mtrx-hub-dev-3of.release_${bq_release_version}.ground_truth_edges_normalized`        
            JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.rtx_kg2_nodes_normalized` ON object = id
        WHERE object_normalization_success = false
    )    
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
    row_num <= 100
