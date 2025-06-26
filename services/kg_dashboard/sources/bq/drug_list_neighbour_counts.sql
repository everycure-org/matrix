WITH Drugs AS (
  SELECT 
    id
  FROM 
    `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized`
)

, Drugs_Neighbours AS (
  SELECT
    Drugs.*
    , object AS neighbour_id
  FROM Drugs
  LEFT JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.edges_unified` drug_is_subject_edges
  ON drug_is_subject_edges.subject = Drugs.id

  UNION ALL

  SELECT
    Drugs.*
    , subject AS neighbour_id
  FROM Drugs
  LEFT JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.edges_unified` drug_is_object_edges
  ON drug_is_object_edges.object = Drugs.id

)

SELECT
  id
  , COUNT(DISTINCT neighbour_id) AS unique_neighbours
FROM 
  Drugs_Neighbours
WHERE 
  id != neighbour_id
GROUP BY 
  id
ORDER BY 
  unique_neighbours