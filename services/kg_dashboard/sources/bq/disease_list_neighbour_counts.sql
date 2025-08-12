WITH Diseases AS (
  SELECT 
    id
  FROM 
    `${project_id}.release_${bq_release_version}.disease_list_nodes_normalized`
)

, Diseases_Neighbours AS (
  SELECT
    Diseases.*
    , object AS neighbour_id
  FROM 
    Diseases
    LEFT JOIN `${project_id}.release_${bq_release_version}.edges_unified` disease_is_subject_edges ON disease_is_subject_edges.subject = Diseases.id

  UNION ALL

  SELECT
    Diseases.*
    , subject AS neighbour_id
  FROM 
    Diseases
    LEFT JOIN `${project_id}.release_${bq_release_version}.edges_unified` disease_is_object_edges ON disease_is_object_edges.object = Diseases.id

)

SELECT
  id
  , COUNT(DISTINCT neighbour_id) AS unique_neighbours
FROM 
  Diseases_Neighbours
WHERE 
  id != neighbour_id
GROUP BY 
  id
ORDER BY 
  unique_neighbours