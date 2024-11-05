SELECT SPLIT(subject, ':')[OFFSET(0)] AS subject_prefix,  
       subject_nodes.category AS subject_category,
       predicate, 
       SPLIT(object, ':')[OFFSET(0)] AS object_prefix,
       object_nodes.category AS object_category,   
       primary_knowledge_source,       
       ARRAY_TO_STRING(edges.upstream_data_source, ', ') AS upstream_data_source,
       count(*) AS count
from `mtrx-hub-dev-3of.release_v0_2_2.edges` AS edges
  JOIN `mtrx-hub-dev-3of.release_v0_2_2.nodes` AS subject_nodes ON edges.subject = subject_nodes.id
  JOIN `mtrx-hub-dev-3of.release_v0_2_2.nodes` AS object_nodes ON edges.object = object_nodes.id
GROUP BY all
