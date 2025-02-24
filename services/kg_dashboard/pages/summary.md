---
title: Summary
---

<!-- TODO: rename merged_kg_nodes sources as misleading  -->
```sql edges_per_node
select 
    all_edges / all_nodes as edges_per_node
    , edges_without_hyperconnected_nodes / nodes_without_hyperconnected_nodes as edges_per_node_without_hyperconnected_nodes
    , disease_edges / disease_nodes as disease_edges_per_node
    , drug_edges / drug_nodes as drug_edges_per_node
from 
    bq.overall_metrics
```

# Edge density
<br/>

<Grid col=2>
    <p class="text-center text-lg"><span class="font-semibold text-4xl"><Value data={edges_per_node} column="edges_per_node" /></span><br/>edges per node on average</p>
    <p class="text-center text-lg"><span class="font-semibold text-4xl"><Value data={edges_per_node} column="edges_per_node_without_hyperconnected_nodes" /></span><br/>edges per node excluding the top 1,000 most connected nodes</p>
</Grid>

# Disease list connections
<br/>

<p class="text-center text-lg"><span class="font-semibold text-4xl"><Value data={edges_per_node} column="disease_edges_per_node" /></span><br/>edges per disease node on average</p>

```sql disease_list_connected_categories
with total as (
    select 
        sum(c) as total_sum
    from 
        bq.disease_list_connected_categories
)

, cumulative_sum as (
    select 
        category
        , c
        , 100.0 * sum(c) over (order by c desc) / total_sum as cumsum_percentage
    from 
        bq.disease_list_connected_categories, total
)

select 
    category
    , c as number_of_connections
from 
    cumulative_sum
where 
    -- TODO: parameterize this 
    cumsum_percentage <= 98.0
order by 
    c desc
```

<BarChart 
    data={disease_list_connected_categories} 
    x="category" 
    y="number_of_connections" 
    swapXY=true
/>

# Drug list connections
<br/>

<p class="text-center text-lg"><span class="font-semibold text-4xl"><Value data={edges_per_node} column="drug_edges_per_node" /></span><br/>edges per drug node on average</p>

```sql drug_list_connected_categories
with total as (
    select 
        sum(c) as total_sum
    from 
        bq.drug_list_connected_categories
)

, cumulative_sum as (
    select 
        category
        , c
        , 100.0 * sum(c) over (order by c desc) / total_sum as cumsum_percentage
    from 
        bq.drug_list_connected_categories, total
)

select 
    category
    , c as number_of_connections
from 
    cumulative_sum
where 
    -- TODO: parameterize this 
    cumsum_percentage <= 98.0
order by 
    c desc
```

<BarChart 
    data={drug_list_connected_categories} 
    x="category" 
    y="number_of_connections" 
    swapXY=true
/>