with drug_out_edges as (
  select 
    subject, object
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.edges` e 
    inner join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` di on e.subject = di.id
)

, drug_in_edges as (
  select 
    subject, object
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.edges` e 
    inner join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` di on e.object = di.id
)

, drug_connections as (
  select object as id from drug_out_edges
  union all 
  select subject as id from drug_in_edges
)

select 
  replace(n.category, "biolink:", "") as category
  , count(distinct d.id) as n_nodes
  , count(*) as n_connections
from
  drug_connections d
  inner join `mtrx-hub-dev-3of.release_${bq_release_version}.nodes` n on d.id = n.id 
group by 1
order by 2 desc
