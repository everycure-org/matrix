---
title: Release Trends
---

<script>
  const current_release_version = import.meta.env.VITE_release_version;
  const build_time = import.meta.env.VITE_build_time;
</script>

# Knowledge Graph Release Trends

This page shows how key metrics have changed across all releases of the MATRIX knowledge graph, with detailed comparisons between the current release ({current_release_version}) and previous versions.

<p class="text-gray-500 text-sm italic">Last updated on {build_time}</p>

```sql release_metrics
select 
    semantic_version,
    bq_version,
    is_current_release,
    n_nodes,
    n_edges,
    n_distinct_knowledge_sources,
    edges_per_node,
    n_nodes_from_disease_list,
    n_nodes_from_drug_list,
    median_drug_node_degree,
    median_disease_node_degree,
    nodes_change,
    edges_change,
    nodes_pct_change,
    edges_pct_change
from 
    bq.release_trends
where 
    -- Only include releases that have data (some may not have overall_metrics table)
    n_nodes is not null
    and n_edges is not null
order by 
    release_order
```

## Graph Size

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_nodes"
    title="Total Nodes Across Releases"
    yGridlines=false
    xBaseline=false
    markers=false
    lineColor="#88C0D0"
    lineWidth=3
    sort=false
/>

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_edges"
    title="Total Edges Across Releases" 
    yGridlines=false
    xBaseline=false
    markers=false
    lineColor="#9D79D6"
    lineWidth=3
    sort=false
/>

</div>

## Drug List

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_nodes_from_drug_list"
    title="Drug Nodes Across Releases"
    yGridlines=false
    xBaseline=false
    markers=false
    lineColor="#73C991"
    lineWidth=3
    sort=false
/>

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="median_drug_node_degree"
    title="Median Drug Node Degree"
    yGridlines=false
    xBaseline=false
    markers=false
    lineColor="#6FAF8C"
    lineWidth=3
    sort=false
/>

</div>

## Disease List

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_nodes_from_disease_list"
    title="Disease Nodes Across Releases" 
    yGridlines=false
    xBaseline=false
    markers=false
    lineColor="#BF616A"
    lineWidth=3
    sort=false
/>

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="median_disease_node_degree"
    title="Median Disease Node Degree"
    yGridlines=false
    xBaseline=false
    markers=false
    lineColor="#D08770"
    lineWidth=3
    sort=false
/>

</div>

## Knowledge Sources

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_distinct_knowledge_sources"
    title="Number of primary knowledge sources in each release"
    yGridlines=false
    xBaseline=false
    markers=false
    lineColor="#7FADDB"
    lineWidth=3
    sort=false
/>


