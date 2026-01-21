-- List of Knowledge Graphs with exclusions and display name mapping
SELECT DISTINCT
  upstream_data_source as knowledge_graph,
  CASE upstream_data_source
    WHEN 'rtx-kg2' THEN 'RTX-KG2'
    WHEN 'robokop' THEN 'ROBOKOP'
    WHEN 'primekg' THEN 'PrimeKG'
    ELSE INITCAP(REPLACE(upstream_data_source, '_', ' '))
  END as display_name
FROM `${project_id}.release_${bq_release_version}.unified_normalization_summary`
WHERE upstream_data_source NOT IN (
  'disease_list',           -- EC Core Entity
  'drug_list',              -- EC Core Entity
  'ec_ground_truth',        -- Evaluation data
  'ec_clinical_trials',     -- Internal data
  'kgml_xdtd_ground_truth', -- Evaluation data
  'off_label'               -- Internal data
)
ORDER BY upstream_data_source
