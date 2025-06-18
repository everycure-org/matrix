with nodes_per_source as (
  select 
     sources.element as upstream_data_source
    , count(*) as c 
  from 
    `${project_id}.release_${bq_release_version}.nodes_unified` n
    , unnest(upstream_data_source.list) as sources
  group by 1 
  order by 2 desc
)

, edges_per_source as (
  select 
     sources.element as upstream_data_source
    , count(*) as c 
  from 
    `${project_id}.release_${bq_release_version}.edges_unified` n
    , unnest(upstream_data_source.list) as sources
  group by 1 
  order by 2 desc
)

select 
  coalesce(n.upstream_data_source, e.upstream_data_source) as upstream_data_source
  , sum(coalesce(n.c, 0)) as n_nodes
  , sum(coalesce(e.c, 0)) as n_edges
from
  nodes_per_source n
  full outer join edges_per_source e on n.upstream_data_source = e.upstream_data_source
group by 1
order by 2 desc 