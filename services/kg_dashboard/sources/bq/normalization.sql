SELECT 
  SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
  SPLIT(id, ':')[OFFSET(0)] AS prefix,
  category,
  normalization_success, 
  no_normalization_change, 
  'rtx_kg2' AS normalization_set,
  count(*) AS count
FROM (
  SELECT 
    subject AS id,
    original_subject AS original_id,
    category,
    subject_normalization_success AS normalization_success,
    CASE WHEN original_subject = subject THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.rtx_kg2_edges_normalized`
    JOIN `${project_id}.release_${bq_release_version}.rtx_kg2_nodes_normalized` ON subject = id
  UNION DISTINCT
  SELECT 
    object AS id,
    original_object AS original_id,
    category, 
    object_normalization_success AS normalization_success,
    CASE WHEN original_object = object THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.rtx_kg2_edges_normalized`
    JOIN `${project_id}.release_${bq_release_version}.rtx_kg2_nodes_normalized` ON object = id
)
GROUP BY ALL

UNION ALL

SELECT 
  SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
  SPLIT(id, ':')[OFFSET(0)] AS prefix,
  category,
  normalization_success, 
  no_normalization_change, 
  'robokop' AS normalization_set,
  count(*) AS count
FROM (
  SELECT 
    subject AS id,
    original_subject AS original_id,
    category,
    subject_normalization_success AS normalization_success,
    CASE WHEN original_subject = subject THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.robokop_edges_normalized`
    JOIN `${project_id}.release_${bq_release_version}.robokop_nodes_normalized` ON subject = id
  UNION DISTINCT
  SELECT 
    object AS id,
    original_object AS original_id, 
    category,
    object_normalization_success AS normalization_success,
    CASE WHEN original_object = object THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.robokop_edges_normalized`
    JOIN `${project_id}.release_${bq_release_version}.robokop_nodes_normalized` ON object = id
)
GROUP BY ALL

UNION ALL

SELECT 
  SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
  SPLIT(id, ':')[OFFSET(0)] AS prefix,
  category,
  normalization_success,
  CASE WHEN original_id = id THEN true ELSE false END AS no_normalization_change,
  'ground_truth' AS normalization_set,
  count(*) AS count
FROM (
  SELECT 
    subject AS id,
    original_subject AS original_id,
    '' as category,
    subject_normalization_success AS normalization_success,
    CASE WHEN original_subject = subject THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.kgml_xdtd_ground_truth_edges_normalized`
  UNION DISTINCT
  SELECT 
    object AS id,
    original_object AS original_id,
    '' as category,
    object_normalization_success AS normalization_success,
    CASE WHEN original_object = object THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.kgml_xdtd_ground_truth_edges_normalized`
  UNION DISTINCT
  SELECT 
    subject AS id,
    original_subject AS original_id,
    '' as category,
    subject_normalization_success AS normalization_success,
    CASE WHEN original_subject = subject THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.ec_ground_truth_edges_normalized`
  UNION DISTINCT
  SELECT 
    object AS id,
    original_object AS original_id,
    '' as category,
    object_normalization_success AS normalization_success,
    CASE WHEN original_object = object THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.ec_ground_truth_edges_normalized`
)
GROUP BY ALL

UNION ALL

SELECT
  SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
  SPLIT(id, ':')[OFFSET(0)] AS prefix,
  category,
  normalization_success, 
  CASE WHEN original_id = id THEN true ELSE false END AS no_normalization_change,
  'drug_list' AS normalization_set,
  count(*) AS count
  FROM 
    `${project_id}.release_${bq_release_version}.drug_list_nodes_normalized`  
  WHERE id <> "['Error']"
GROUP BY ALL

UNION ALL

SELECT
  SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
  SPLIT(id, ':')[OFFSET(0)] AS prefix,
  category,
  normalization_success, 
  CASE WHEN original_id = id THEN true ELSE false END AS no_normalization_change,
  'disease_list' AS normalization_set,
  count(*) AS count
  FROM 
    `${project_id}.release_${bq_release_version}.disease_list_nodes_normalized`  
GROUP BY ALL

UNION ALL

SELECT 
  SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
  SPLIT(id, ':')[OFFSET(0)] AS prefix,
  category,
  normalization_success, 
  no_normalization_change, 
  'merged' AS normalization_set,
  count(*) AS count
FROM (
  SELECT 
    subject AS id,
    original_subject AS original_id,
    category,
    subject_normalization_success AS normalization_success,
    CASE WHEN original_subject = subject THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.rtx_kg2_edges_normalized`
    JOIN `${project_id}.release_${bq_release_version}.rtx_kg2_nodes_normalized` ON subject = id
  UNION DISTINCT
  SELECT 
    object AS id, 
    original_object AS original_id,
    category,
    object_normalization_success AS normalization_success,
    CASE WHEN original_object = object THEN true ELSE false END AS no_normalization_change 
  FROM 
    `${project_id}.release_${bq_release_version}.rtx_kg2_edges_normalized`
    JOIN `${project_id}.release_${bq_release_version}.rtx_kg2_nodes_normalized` ON object = id
  UNION DISTINCT
  SELECT 
    subject AS id,
    original_subject AS original_id,
    category,
    subject_normalization_success AS normalization_success,
    CASE WHEN original_subject = subject THEN true ELSE false END AS no_normalization_change
  FROM 
    `${project_id}.release_${bq_release_version}.robokop_edges_normalized`
    JOIN `${project_id}.release_${bq_release_version}.robokop_nodes_normalized` ON subject = id
  UNION DISTINCT
  SELECT 
    object AS id,
    original_object AS original_id,
    category,
    object_normalization_success AS normalization_success,
    CASE WHEN original_object = object THEN true ELSE false END AS no_normalization_change
  FROM 
    `${project_id}.release_${bq_release_version}.robokop_edges_normalized`
    JOIN `${project_id}.release_${bq_release_version}.robokop_nodes_normalized` ON object = id
  UNION DISTINCT
  SELECT 
    subject AS id,
    original_subject AS original_id,
    '' AS category,
    subject_normalization_success AS normalization_success,
    CASE WHEN original_subject = subject THEN true ELSE false END AS no_normalization_change
  FROM 
    `${project_id}.release_${bq_release_version}.kgml_xdtd_ground_truth_edges_normalized`        
  UNION DISTINCT
  SELECT 
    object AS id,
    original_object AS original_id,
    '' AS category,
    object_normalization_success AS normalization_success,
    CASE WHEN original_object = object THEN true ELSE false END AS no_normalization_change
  FROM 
    `${project_id}.release_${bq_release_version}.kgml_xdtd_ground_truth_edges_normalized`        
  UNION DISTINCT
  SELECT 
    subject AS id,
    original_subject AS original_id,
    '' AS category,
    subject_normalization_success AS normalization_success,
    CASE WHEN original_subject = subject THEN true ELSE false END AS no_normalization_change
  FROM 
    `${project_id}.release_${bq_release_version}.ec_ground_truth_edges_normalized`        
  UNION DISTINCT
  SELECT 
    object AS id,
    original_object AS original_id,
    '' AS category,
    object_normalization_success AS normalization_success,
    CASE WHEN original_object = object THEN true ELSE false END AS no_normalization_change
  FROM 
    `${project_id}.release_${bq_release_version}.ec_ground_truth_edges_normalized`        
  UNION DISTINCT
  SELECT
    id,
    original_id,
    category,
    normalization_success,
    CASE WHEN original_id = id THEN true ELSE false END AS no_normalization_change
  FROM
    `${project_id}.release_${bq_release_version}.drug_list_nodes_normalized`
  UNION DISTINCT
  SELECT
    id,
    original_id,
    category,
    normalization_success,
    CASE WHEN original_id = id THEN true ELSE false END AS no_normalization_change
  FROM
    `${project_id}.release_${bq_release_version}.disease_list_nodes_normalized`
)
GROUP BY ALL
