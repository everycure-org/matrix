SELECT
  CONCAT('infores:', pks_id) AS primary_knowledge_source,
  domain_coverage_score,
  domain_coverage_comments,
  source_scope_score,
  source_scope_score_comment,
  utility_drugrepurposing_score,
  utility_drugrepurposing_comment,
  label_rubric,
  label_manual,
  reviewer
FROM `${project_id}.release_${bq_release_version}.all_pks_metadata_table`
WHERE domain_coverage_score IS NOT NULL
   OR source_scope_score IS NOT NULL
   OR utility_drugrepurposing_score IS NOT NULL
