-- =====
-- Get the ids of the most connected nodes in the KG i.e. the nodes with the highest degree
-- =====
with edge_count_per_subject as (
  select 
    subject, count(*) as n_edges_per_subject
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.edges_unified`
  group by 1
)

, edge_count_per_object as (
  select 
    object, count(*) as n_edges_per_object
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.edges_unified`
  where 
    -- Self edges are already captured in "edge_count_per_subject", we remove them here as we don't want to count them twice 
    object != subject 
  group by 1
)

, edge_count as (
  select 
    subject as id
    , n_edges_per_subject as n_edges 
  from 
    edge_count_per_subject s 

  union all 
  
  select 
    object as id
    , n_edges_per_object as n_edges 
  from 
    edge_count_per_object o
)

-- sum the number of edges of each node to obtain its degree
, nodes_degrees as (
  select 
    id
    , sum(n_edges) as degree
  from 
    edge_count
  group by 1
)

, most_connected_nodes as (
  select 
    *
  from 
    nodes_degrees
  order by 
    degree desc 
  -- We only want to remove the top 1000 most connected nodes. This number was chosen after analysis of the KG nodes' degrees distribution.
  -- TODO: find a way to parameterize this number in BigQuery
  limit 1000
)

, median_drug_node_degree as (
  select 
    PERCENTILE_CONT(n.degree, 0.5) OVER() as median
  from 
    nodes_degrees n
    inner join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dr on n.id = dr.id
  limit 1
)

, median_disease_node_degree as (
  select 
    PERCENTILE_CONT(n.degree, 0.5) OVER() as median
  from 
    nodes_degrees n
    inner join `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` di on n.id = di.id
  limit 1
)

-- =====
-- Count nodes and edges 
-- =====

, n_nodes as (
  select 
    count(*) as n_nodes
    , SUM(IF(m.id is null, 1, 0)) as n_nodes_without_most_connected_nodes
    , SUM(IF(di.id is not null, 1, 0)) as n_nodes_from_disease_list
    , SUM(IF(dr.id is not null, 1, 0)) as n_nodes_from_drug_list
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.nodes_unified` n
    left outer join most_connected_nodes m on n.id = m.id
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` di on n.id = di.id 
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dr on n.id = dr.id 
)

, n_edges as (
  select 
    count(*) as n_edges
    , SUM(IF(m1.id is null and m2.id is null, 1, 0)) as n_edges_without_most_connected_nodes
    , SUM(IF(di1.id is not null or di2.id is not null, 1, 0)) as n_edges_from_disease_list
    , SUM(IF(dr1.id is not null or dr2.id is not null, 1, 0)) as n_edges_from_drug_list
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.edges_unified` e
    left outer join most_connected_nodes m1 on e.subject = m1.id
    left outer join most_connected_nodes m2 on e.object = m2.id
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` di1 on e.subject = di1.id 
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized` di2 on e.object = di2.id 
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dr1 on e.subject = dr1.id 
    left outer join `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` dr2 on e.object = dr2.id 
)

, output_metrics as (
  select 
    n_nodes.n_nodes
    , n_nodes.n_nodes_without_most_connected_nodes
    , n_nodes.n_nodes_from_disease_list
    , n_nodes.n_nodes_from_drug_list
    , n_edges.n_edges
    , n_edges.n_edges_without_most_connected_nodes
    , n_edges.n_edges_from_disease_list
    , n_edges.n_edges_from_drug_list
    , median_drug_node_degree.median as median_drug_node_degree
    , median_disease_node_degree.median as median_disease_node_degree
  from 
    n_nodes 
    cross join n_edges
    cross join median_drug_node_degree
    cross join median_disease_node_degree
)

select 
  *
from
  output_metrics