WITH drug_list AS (

  SELECT 
    curie AS drug_id,
    curie_label AS drug_name
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized`

),

joined_list AS (
  SELECT
    drug_list.*,
    object AS neighbour_id
  FROM drug_list
  LEFT JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.edges` drug_is_subject_edges
  ON drug_is_subject_edges.subject = drug_list.drug_id

  UNION ALL

  SELECT
    drug_list.*,
    subject AS neighbour_id
  FROM drug_list
  LEFT JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.edges` drug_is_object_edges
  ON drug_is_object_edges.object = drug_list.drug_id

)

SELECT
  drug_id,
  drug_name,
  COUNT(DISTINCT neighbour_id) AS unique_neighbours
FROM joined_list
WHERE drug_id != neighbour_id
GROUP BY drug_id, drug_name ORDER BY unique_neighbours;