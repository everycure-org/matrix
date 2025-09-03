---
title: KG Dashboard
---

<script context="module">
    import { getSourceColor } from './_lib/colors';
    
    // Function to get colors for pie chart data
    export function getPieColors(data) {
        return data.map(item => getSourceColor(item.name));
    }

    const release_version = import.meta.env.VITE_release_version;
    const build_time = import.meta.env.VITE_build_time;
    const robokop_version = import.meta.env.VITE_robokop_version;
    const rtx_kg2_version = import.meta.env.VITE_rtx_kg2_version;
    const benchmark_version = import.meta.env.VITE_benchmark_version;

</script>


<div class="mb-4 flex flex-col gap-2">
    <a href="https://docs.dev.everycure.org/releases/release_history/" class="inline-flex items-center text-blue-600 hover:text-blue-800 text-sm">
        ← Release History 
    </a>
    <a href="https://data.dev.everycure.org/versions/{benchmark_version}/evidence/" target="_blank" class="inline-flex items-center text-blue-600 hover:text-blue-800 text-sm">
        🔖 Benchmark Release ({benchmark_version})
    </a>
</div>

This dashboard provides an overview of our integrated knowledge graph (KG), detailing its size, connectivity patterns, and provenance quality. 
It also examines how nodes from our curated disease and drug lists link to other entities within the graph.
## Version: {release_version}

<p class="text-gray-500 text-sm italic">Last updated on {build_time}</p>

```sql edges_per_node
select 
    n_nodes
    , n_edges
    , n_edges / n_nodes as edges_per_node
    , n_edges_without_most_connected_nodes / n_nodes_without_most_connected_nodes as edges_per_node_without_most_connected_nodes
    , n_edges_from_disease_list / n_nodes_from_disease_list as disease_edges_per_node
    , n_edges_from_drug_list / n_nodes_from_drug_list as drug_edges_per_node
    , median_drug_node_degree
    , median_disease_node_degree
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


## Upstream data sources 

```sql upstream_data_sources_nodes
select 
    upstream_data_source as name
    , n_nodes as value
from 
    bq.upstream_data_sources   
```

```sql upstream_data_sources_edges
select 
    upstream_data_source as name
    , n_edges as value
from 
    bq.upstream_data_sources   
```
<Details title="Source details">
  <div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mt-2">
    This release integrates nodes and edges from multiple upstream sources, shown in the charts below. 
    The versions listed indicate the specific snapshots used for this build of the knowledge graph.<br/>
    <br><strong>Knowledge Graph Versions:</strong><br/>
    • <strong>ROBOKOP:</strong> <span class="font-mono">{robokop_version}</span> <br/>
    • <strong>RTX-KG2:</strong> <span class="font-mono">{rtx_kg2_version}</span> <br/>
   
  </div>
</Details>

<Grid col=2>
    <ECharts 
        config={{
            title: {
                text: 'Nodes',
                left: 'center',
                top: 'center',
                textStyle: {
                    fontWeight: 'normal'
                }
            },
            color: getPieColors(upstream_data_sources_nodes),
            tooltip: {
                formatter: function(params) {
                    const count = params.data.value.toLocaleString();
                    return `${params.name}: ${count} nodes (${params.percent}%)`;
                }
            },
            series: [{
                type: 'pie', 
                data: [...upstream_data_sources_nodes],
                radius: ['30%', '50%'],
            }]
        }}
    />
    <ECharts config={{
        title: {
            text: 'Edges',
            left: 'center',
            top: 'center',
            textStyle: {
                fontWeight: 'normal'
            }
        },
        color: getPieColors(upstream_data_sources_edges),
        tooltip: {
            formatter: function(params) {
                const count = params.data.value.toLocaleString();
                return `${params.name}: ${count} edges (${params.percent}%)`;
            }
        },
        series: [{
            type: 'pie', 
            data: [...upstream_data_sources_edges],
            radius: ['30%', '50%'],
        }]
    }}/>
</Grid>

## Disease list connections

<Details title="Click to Understand This Chart">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  This section summarizes the connectivity of nodes in the disease list, measured by their number of direct neighbors in the 
  knowledge graph. The mean and median values reflect how many other entities (such as drugs, genes, or phenotypes) 
  each disease node is linked to. This helps characterize the typical network context for diseases of interest, and highlights 
  how densely or sparsely connected different parts of the graph may be. For more details visit 
  <a class="underline text-blue-600" href="./EC%20Core%20Entities ">EC Core Entities</a>. 
</div>
</Details>

<br/>

<Grid col=2>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="disease_edges_per_node" fmt="num1"/></span><br/>mean neighbours per disease node</p>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="median_disease_node_degree" fmt="num0"/></span><br/>median neighbours per disease node</p>
</Grid>

<br/>


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
    cumsum_percentage <= 90.0
order by 
    n_connections desc
```

<BarChart 
    data={disease_list_connected_categories} 
    x="category" 
    y="number_of_connections" 
    colorPalette={[getSourceColor("disease_list")]}
    swapXY=true
    title="Categories connected to disease list node on average"
/>

## Drug list connections

<Details title="Click to Understand This Chart">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  This section summarizes the connectivity of nodes in the drug list, measured by their number of direct neighbors in the 
  knowledge graph. The mean and median values reflect how many other entities (such as disease, genes, or phenotypes) 
  each drug node is linked to. This helps characterize the typical network context for drug of interest, and highlights 
  how densely or sparsely connected different parts of the graph may be. For more details visit 
  <a class="underline text-blue-600" href="./EC%20Core%20Entities ">EC Core Entities</a>. 
</div>
</Details>

<br/>

<Grid col=2>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="drug_edges_per_node" fmt="num1"/></span><br/>mean neighbours per drug node</p>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="median_drug_node_degree" fmt="num0"/></span><br/>median neighbours per drug node</p>
</Grid>

<br/>

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
    cumsum_percentage <= 90.0
order by 
    n_connections desc
```

<BarChart 
    data={drug_list_connected_categories} 
    x="category" 
    y="number_of_connections" 
    colorPalette={[getSourceColor("drug_list")]}
    swapXY=true
    title="Categories connected to drug list node on average"
/>