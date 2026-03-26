SELECT
  CASE
    WHEN component_id = 0 THEN 'LCC'
    WHEN component_size >= 100 THEN '100+ nodes'
    WHEN component_size >= 10 THEN '10–99 nodes'
    WHEN component_size >= 2 THEN '2–9 nodes'
    ELSE 'Isolated (1 node)'
  END as size_category,
  CASE
    WHEN component_id = 0 THEN 1
    WHEN component_size >= 100 THEN 2
    WHEN component_size >= 10 THEN 3
    WHEN component_size >= 2 THEN 4
    ELSE 5
  END as sort_order,
  COUNT(*) as num_components,
  SUM(component_size) as total_nodes,
  SUM(num_drugs) as core_drugs,
  SUM(num_diseases) as core_diseases,
  SUM(num_other) as other_nodes
FROM `${project_id}.release_${bq_release_version}.connected_components`
GROUP BY size_category, sort_order
ORDER BY sort_order
