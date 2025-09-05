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

## Graph Size Trends

### Nodes and Edges Over Time

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_nodes"
    title="Total Nodes Across Releases"
    yAxisTitle="Number of Nodes"
    sort=false
/>

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_edges"
    title="Total Edges Across Releases" 
    yAxisTitle="Number of Edges"
    sort=false
/>

## Drug and Disease lists over time

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_nodes_from_drug_list"
    title="Drug Nodes Across Releases"
    yAxisTitle="Number of Drug Nodes"
    sort=false
/>

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_nodes_from_disease_list"
    title="Disease Nodes Across Releases" 
    yAxisTitle="Number of Disease Nodes"
    sort=false
/>

## Knowledge Sources

<LineChart 
    data={release_metrics} 
    x="semantic_version" 
    y="n_distinct_knowledge_sources"
    title="Distinct Knowledge Sources Across Releases"
    yAxisTitle="Number of Distinct Knowledge Sources"
    sort=false
/>


