WITH disease_list AS (

  SELECT 
    category_class AS disease_id,
    label AS disease_name
  FROM `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized`

),

joined_list AS (
  SELECT
    disease_list.*,
    object AS neighbour_id
  FROM disease_list
  LEFT JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.edges` disease_is_subject_edges
  ON disease_is_subject_edges.subject = disease_list.disease_id

  UNION ALL

  SELECT
    disease_list.*,
    subject AS neighbour_id
  FROM disease_list
  LEFT JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.edges` disease_is_object_edges
  ON disease_is_object_edges.object = disease_list.disease_id

)

SELECT
  disease_id,
  disease_name,
  COUNT(DISTINCT neighbour_id) AS unique_neighbours
FROM joined_list
WHERE disease_id != neighbour_id
GROUP BY disease_id, disease_name ORDER BY unique_neighbours;