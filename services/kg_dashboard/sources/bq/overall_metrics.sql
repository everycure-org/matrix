with edge_count_per_subject as (
  select 
    subject, count(*) as c 
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.edges`
  group by 1
)

, edge_count_per_object as (
  select 
    object, count(*) as c 
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.edges`
  where 
    -- we don't want to count edges to self twice
    object != subject 
  group by 1
)

, most_connected_nodes as (
  select 
    coalesce(subject, object) as node_id 
    -- , coalesce(s.c, 0) + coalesce(o.c, 0) as n_edges
  from 
    edge_count_per_subject s 
    full outer join edge_count_per_object o on s.subject = o.object
  order by 
    coalesce(s.c, 0) + coalesce(o.c, 0) desc 
  limit 1000
)

, n_nodes as (
  select 
    count(*) as all_nodes
    , SUM(IF(m.node_id is null, 1, 0)) as nodes_without_hyperconnected_nodes
    , SUM(IF(di.id is not null, 1, 0)) as disease_nodes
    , SUM(IF(dr.id is not null, 1, 0)) as drug_nodes
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.nodes` n
    left outer join most_connected_nodes m on n.id = m.node_id
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` di on n.id = di.id 
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dr on n.id = dr.id 
)

, n_edges as (
  select 
    count(*) as all_edges
    , SUM(IF(m1.node_id is null and m2.node_id is null, 1, 0)) as edges_without_hyperconnected_nodes
    , SUM(IF(di1.id is not null or di2.id is not null, 1, 0)) as disease_edges
    , SUM(IF(dr1.id is not null or dr2.id is not null, 1, 0)) as drug_edges
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.edges` e
    left outer join most_connected_nodes m1 on e.subject = m1.node_id
    left outer join most_connected_nodes m2 on e.object = m2.node_id
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` di1 on e.subject = di1.id 
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` di2 on e.object = di2.id 
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dr1 on e.subject = dr1.id 
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dr2 on e.object = dr2.id 
)

, output_metrics as (
  select 
    n_nodes.all_nodes
    , n_nodes.nodes_without_hyperconnected_nodes
    , n_nodes.disease_nodes
    , n_nodes.drug_nodes
    , n_edges.all_edges
    , n_edges.edges_without_hyperconnected_nodes
    , n_edges.disease_edges
    , n_edges.drug_edges
  from 
    n_nodes 
    cross join n_edges
)

select 
  *
from
  output_metrics