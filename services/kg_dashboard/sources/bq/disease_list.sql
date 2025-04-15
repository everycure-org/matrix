SELECT disease_list.*, count(*) AS edge_count
FROM `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` as disease_list  
  JOIN `mtrx-hub-dev-3of.release_${bq_release_version}.edges` AS edges
    ON disease_list.id = subject or disease_list.id = object
GROUP BY all
