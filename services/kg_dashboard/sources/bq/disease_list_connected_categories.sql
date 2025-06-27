with disease_out_edges as (
  select 
    subject, object
  from 
    `${project_id}.release_${bq_release_version}.edges_unified` e 
    inner join `${project_id}.release_${bq_release_version}.disease_list_nodes_normalized` di on e.subject = di.id
)

, disease_in_edges as (
  select 
    subject, object
  from 
    `${project_id}.release_${bq_release_version}.edges_unified` e 
    inner join `${project_id}.release_${bq_release_version}.disease_list_nodes_normalized` di on e.object = di.id
)

, disease_connections as (
  select object as id from disease_out_edges
  union all 
  select subject as id from disease_in_edges
)

select 
  replace(n.category, "biolink:", "") as category
  , count(*) as n_connections
from
  disease_connections d
  inner join `${project_id}.release_${bq_release_version}.nodes_unified` n on d.id = n.id 
group by 1
order by 2 desc
