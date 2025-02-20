SELECT 
       SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
       SPLIT(id, ':')[OFFSET(0)] AS prefix,
       category,
       case when original_id = id then true else false end as no_normalization_change,
       normalization_success,
       'disease_list' as normalization_set,
       count(*) as count
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_disease_list`
GROUP BY all
UNION DISTINCT
SELECT 
       SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
       SPLIT(id, ':')[OFFSET(0)] AS prefix,       
       category,
       case when original_id = id then true else false end as no_normalization_change,
       normalization_success,
       'drug_list' as normalization_set,
       count(*) as count
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_drug_list`
GROUP BY all
UNION DISTINCT
SELECT 
       SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
       SPLIT(id, ':')[OFFSET(0)] AS prefix,
       null as category, -- ec_clinical_trials does not have a category column
       case when original_id = id then true else false end as no_normalization_change,
       normalization_success,
       'ec_clinical_trials' as normalization_set,
       count(*) as count
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_ec_clinical_trails`
GROUP BY all
UNION DISTINCT
SELECT 
       SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
       SPLIT(id, ':')[OFFSET(0)] AS prefix,
       null as category, -- ground_truth does not have a category column
       case when original_id = id then true else false end as no_normalization_change,
       normalization_success,
       'ground_truth' as normalization_set,
       count(*) as count
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_ground_truth`
GROUP BY all
UNION DISTINCT
SELECT 
       SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
       SPLIT(id, ':')[OFFSET(0)] AS prefix,
       category,
       case when original_id = id then true else false end as no_normalization_change,
       normalization_success,
       'rtx_kg2' as normalization_set,
       count(*) as count
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_rtx_kg2`
GROUP BY all
UNION DISTINCT
SELECT 
       SPLIT(original_id, ':')[OFFSET(0)] AS original_prefix,
       SPLIT(id, ':')[OFFSET(0)] AS prefix,
       category,
       case when original_id = id then true else false end as no_normalization_change,
       normalization_success as normalization_set,
       'merged' as normalization_set,
       count(*) as count
FROM (
  SELECT original_id, id, category, normalization_success 
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_disease_list`
  UNION DISTINCT
  SELECT original_id, id, category, normalization_success 
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_drug_list`
  UNION DISTINCT
  SELECT original_id, id, null as category, normalization_success 
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_ec_clinical_trails`
  UNION DISTINCT
  SELECT original_id, id, null as category, normalization_success 
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_ground_truth`
  UNION DISTINCT
  SELECT original_id, id, category, normalization_success 
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_normalized_rtx_kg2`
)
GROUP BY ALL
;