SELECT SPLIT(subject, ':')[OFFSET(0)] AS subject_prefix,  
       subject_nodes.category AS subject_category,
       predicate, 
       SPLIT(object, ':')[OFFSET(0)] AS object_prefix,
       object_nodes.category AS object_category,   
       primary_knowledge_source,       
       STRING_AGG(DISTINCT edges.upstream_data_source.list[SAFE_OFFSET(0)].element, ',' ORDER BY edges.upstream_data_source.list[SAFE_OFFSET(0)].element) AS upstream_data_source,
       count(*) AS count
from `mtrx-hub-dev-3of.release_${release_version}.edges` AS edges
  JOIN `mtrx-hub-dev-3of.release_${release_version}.nodes` AS subject_nodes ON edges.subject = subject_nodes.id
  JOIN `mtrx-hub-dev-3of.release_${release_version}.nodes` AS object_nodes ON edges.object = object_nodes.id
GROUP BY all

