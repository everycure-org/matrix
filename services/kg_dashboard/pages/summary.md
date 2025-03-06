---
title: Summary
---

This page provides key metrics about our knowledge graph (KG), including its size, density, and connectivity patterns, with a focus on how nodes from our disease and drug lists are connected within the graph.

```sql edges_per_node
select 
    n_nodes
    , n_edges
    , n_edges / n_nodes as edges_per_node
    , n_edges_without_most_connected_nodes / n_nodes_without_most_connected_nodes as edges_per_node_without_most_connected_nodes
    , n_edges_from_disease_list / n_nodes_from_disease_list as disease_edges_per_node
    , n_edges_from_drug_list / n_nodes_from_drug_list as drug_edges_per_node
from 
    bq.overall_metrics
```

## Graph size

<Grid col=2>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="n_nodes" fmt="num2m"/></span><br/>nodes</p>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="n_edges" fmt="num2m"/></span><br/>edges</p>
</Grid>

<br/>

## Graph density

<Grid col=2>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="edges_per_node" fmt="num1"/></span><br/>edges per node on average</p>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="edges_per_node_without_most_connected_nodes" fmt="num1"/></span><br/>edges per node when excluding the top 1,000 most connected nodes</p>
</Grid>

## Disease list nodes connections

```sql disease_list_connected_categories
with total as (
    select 
        sum(n_connections) as sum_n_connections
    from 
        bq.disease_list_connected_categories
)

, cumulative_sum as (
    select 
        category
        , n_connections
        , 100.0 * sum(n_connections) over (order by n_connections desc) / sum_n_connections as cumsum_percentage
    from 
        bq.disease_list_connected_categories
        , total
)

select 
    category
    , (n_edges_from_disease_list / n_nodes_from_disease_list) * (n_connections / sum_n_connections) as number_of_connections
from 
    cumulative_sum
    , total
    , bq.overall_metrics
where 
    -- TODO: parameterize this 
    cumsum_percentage <= 99.0
order by 
    n_connections desc
```

<br/>

<p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="disease_edges_per_node" fmt="num1"/></span><br/>edges per disease node on average</p>

<br/>

<BarChart 
    data={disease_list_connected_categories} 
    x="category" 
    y="number_of_connections" 
    swapXY=true
    title="Categories connected to disease list node on average"
/>

## Drug list nodes connections

```sql drug_list_connected_categories
with total as (
    select 
        sum(n_connections) as sum_n_connections
    from 
        bq.drug_list_connected_categories
)

, cumulative_sum as (
    select 
        category
        , n_connections
        , 100.0 * sum(n_connections) over (order by n_connections desc) / sum_n_connections as cumsum_percentage
    from 
        bq.drug_list_connected_categories, total
)

select 
    category
    , (n_edges_from_drug_list / n_nodes_from_drug_list) * (n_connections / sum_n_connections) as number_of_connections
from 
    cumulative_sum
    , total
    , bq.overall_metrics
where 
    -- TODO: parameterize this 
    cumsum_percentage <= 99.0
order by 
    n_connections desc
```

<br/>

<p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="drug_edges_per_node" fmt="num1"/></span><br/>edges per drug node on average</p>

<br/>

<BarChart 
    data={drug_list_connected_categories} 
    x="category" 
    y="number_of_connections" 
    swapXY=true
    title="Categories connected to drug list node on average"
/>