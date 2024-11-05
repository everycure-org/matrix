SELECT 
       SPLIT(subject, ':')[OFFSET(0)] AS subject_prefix,  
       predicate, 
       SPLIT(object, ':')[OFFSET(0)] AS object_prefix,  
       knowledge_source AS primary_knowledge_source, 
       'rtx_kg2' AS kg_source, 
       count(*) AS count 
FROM `mtrx-hub-dev-3of.kg_rtx_kg2.edges_20240807` AS edges 
GROUP BY all
UNION all
SELECT SPLIT(subject, ':')[OFFSET(0)] AS subject_prefix,  
       predicate, 
       SPLIT(object, ':')[OFFSET(0)] AS object_prefix,  
       primary_knowledge_source, 
       'robokop_kg' AS kg_source, 
       count(*) AS count 
from `mtrx-hub-dev-3of.robokop_kg.edges_c5ec1f282158182f` AS edges
group by all
