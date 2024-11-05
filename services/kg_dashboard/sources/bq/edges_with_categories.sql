SELECT [subject_nodes.category] AS subject_category,
       SPLIT(subject_nodes.id, ':')[OFFSET(0)] AS subject_prefix,  
       predicate, 
       [object_nodes.category] AS object_category, 
       SPLIT(object_nodes.id, ':')[OFFSET(0)] AS object_prefix,  
       knowledge_source AS primary_knowledge_source, 
       'rtx_kg2' AS kg_source, 
       count(*) AS count 
FROM `mtrx-hub-dev-3of.kg_rtx_kg2.edges_v2_7_3` AS edges 
 JOIN `mtrx-hub-dev-3of.kg_rtx_kg2.nodes_v2_7_3` AS subject_nodes ON edges.subject = subject_nodes.id
 JOIN `mtrx-hub-dev-3of.kg_rtx_kg2.nodes_v2_7_3` AS object_nodes ON edges.object = object_nodes.id
GROUP BY all
UNION all
SELECT subject_nodes.category AS subject_category,
       SPLIT(subject_nodes.id, ':')[OFFSET(0)] AS subject_prefix,  
       predicate, 
       object_nodes.category AS object_category, 
       SPLIT(object_nodes.id, ':')[OFFSET(0)] AS object_prefix,  
       primary_knowledge_source, 
       'robokop_kg' AS kg_source, 
       count(*) AS count 
from `mtrx-hub-dev-3of.robokop_kg.edges_c5ec1f282158182f` AS edges
 JOIN `mtrx-hub-dev-3of.robokop_kg.nodes_c5ec1f282158182f` AS subject_nodes 
   ON edges.subject = subject_nodes.id
 JOIN `mtrx-hub-dev-3of.robokop_kg.nodes_c5ec1f282158182f` AS object_nodes 
   ON edges.object = object_nodes.id
group by all