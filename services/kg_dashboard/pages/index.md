---
title: KG Dashboard
---

<script context="module">
  import { sourceColorMap } from './_lib/colors';
  
  // Function to get colors for pie chart data
  export function getPieColors(data) {
    return data.map(item => sourceColorMap[item.name] || "#6b7280");
  }

  // Create explicit series color mapping
  export function getSeriesColors(data, seriesColumn) {
    const uniqueSources = [...new Set(data.map(row => row[seriesColumn]))];
    
    const seriesColors = {};
    uniqueSources.forEach(source => {
      seriesColors[source] = sourceColorMap[source] || "#6272a4";
    });
    
    return seriesColors;
  }
  
  export function sortBySeries(data, seriesColumn) {
    return data;
  }
</script>

<script>
  const release_version = import.meta.env.VITE_release_version;
  const build_time = import.meta.env.VITE_build_time;
  const robokop_version = import.meta.env.VITE_robokop_version;
  const rtx_kg2_version = import.meta.env.VITE_rtx_kg2_version;

  function groupBy(arr, key) {
    return arr.reduce((acc, item) => {
      const group = item[key];
      acc[group] = acc[group] || [];
      acc[group].push(item);
      return acc;
    }, {});
  }

  // NOTE: This function was partially generated using AI assistance.
  function createHistogramBins(data, binWidth) {
    if (!data || !Array.isArray(data) || data.length === 0) return [];
    
    // Extract the column values
    const values = data.filter(v => v !== null && v !== undefined);
    if (values.length === 0) return [];
    
    // Calculate min and max if not provided
    const min = 0;
    const max = Math.max(...values);
    
    // Calculate number of bins based on width
    const binCount = Math.ceil((max - min) / binWidth);
    const bins = [];
    
    for (let i = 0; i < binCount; i++) {
      const binStart = min + (i * binWidth);
      const binEnd = min + ((i + 1) * binWidth);
      
      const count = values.filter(value => 
        value >= binStart && (i === binCount - 1 ? value <= binEnd : value < binEnd)
      ).length;
      
      bins.push({
        count: count,
        start: binStart,
        end: binEnd
      });
    }
    
    return bins;
  }

  function getHistogramEchartsOptions(data, data_name, data_key, binWidth) {
    const bins = !data || !Array.isArray(data) || data.length === 0 ? [] : createHistogramBins(data.map(d => d[data_key]), binWidth)
    const xAxis = bins.map(d => d.start)
    
    // Create series data with bin labels included
    const seriesData = bins.map(bin => ({
      value: bin.count,
      binStart: bin.start,
      binEnd: bin.end - 1,
    }))
    
    return {
      grid: {
        top: '2%',
        bottom: '20%',
      },
      xAxis: {
        data: xAxis,
        silent: false,
        splitLine: {
          show: false
        },
        splitArea: {
          show: false
        }
      },
      yAxis: {
        splitArea: {
          show: false
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        },
        formatter: function(params) {
          const binStart = params[0].data.binStart;
          const binEnd = params[0].data.binEnd;
          const count = params[0].value;
          return `${count} ${data_name}s have between ${binStart} and ${binEnd} neighbours`;
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 2,
          minValueSpan: 30
        },
        {
          type: 'slider',
          start: 0,
          end: 2,
          minValueSpan: 30
        }
      ],
      series: [
        {
          type: 'bar',
          data: seriesData
        }
      ]
    }
  }

  
</script>

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
### Source versions

<p class="text-center text-md mt-2 mb-6">
  <span class="font-semibold">ROBOKOP KG:</span> <span class="font-mono">{robokop_version}</span> &nbsp; | &nbsp; 
  <span class="font-semibold">RTX-KG2:</span> <span class="font-mono">{rtx_kg2_version}</span>
</p>

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

## Epistemic Robustness

```sql epistemic_score
SELECT * FROM bq.epistemic_score
```

```sql epistemic_heatmap
SELECT * FROM bq.epistemic_heatmap
```

<div class="text-center text-md mt-6 mb-4">
  This section summarizes the overall provenance quality of the knowledge graph by averaging each edge’s knowledge level and agent type.
  Higher scores indicate stronger evidence and manual curation, while lower scores reflect weaker or more speculative data.
  For a detailed breakdown, visit 
  <a class="underline text-blue-600" href="./Metrics/epistemic-robustness" target="_blank">Epistemic Robustness</a>.
</div>

<Grid col=3>
  <div class="text-center">
    <span class="font-semibold text-2xl">
      <Value data={epistemic_score} column="average_epistemic_score" fmt="num2" />
    </span><br/>
    Epistemic Score
  </div>
  <div class="text-center">
    <span class="font-semibold text-2xl">
      <Value data={epistemic_score} column="included_edges" fmt="num2m" />
    </span><br/>
    edges included
  </div>
  <div class="text-center">
    <span class="font-semibold text-2xl">
      <Value data={epistemic_score} column="null_or_not_provided_both" fmt="num2m" />
    </span><br/>
    edges with missing provenance
    <div class="text-sm font-normal mt-1">
      Includes edges where both Knowledge Level and Agent Type are not provided.
    </div>
  </div>
</Grid>

<div class="mt-8 mb-6">
<ECharts
  style={{ width: '100%', height: '2000px' }}
  config={{
    title: {
      text: 'Epistemic Provenance',
      left: 'center',
      top: 10
    },
    tooltip: {
      formatter: function (params) {
        const [x, y, count, score] = params.value;
        return `
          Knowledge Level: ${x}<br/>
          Agent Type: ${y}<br/>
          Edges: ${count.toLocaleString()}<br/>
          Score: ${score.toFixed(2)}
        `;
      }
    },
    grid: {
      top: 30,
      bottom: 100,
      left: 120,
      right: 120,
      containLabel: false
    },
    xAxis: {
      type: 'category',
      data: [
        "Prediction", "Observation", "Not Provided",
        "Statistical Association", "Logical Entailment", "Knowledge Assertion"
      ],
      axisLabel: { rotate: 30, fontSize: 10 }
    },
    yAxis: {
      type: 'category',
      data: [
        "Text Mining Agent", "Image Processing\nAgent", "Not Provided",
        "Computational Model", "Data Analysis Pipeline", "Automated Agent",
        "Manual Validation of\nAutomated Agent", "Manual Agent"
      ],
      axisLabel: { fontSize: 10 }
    },
    visualMap: {
      min: -1.0,
      max: 1.0,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      itemHeight: 400,
      itemWidth: 10,
      precision: 2,
      inRange: {
        color: ['#bd93f9', '#8be9fd', '#50fa7b']
      },
      text: ['Stronger Provenance', 'Weaker Provenance']
    },
    series: [{
      type: 'heatmap',
      label: {
        show: true,
        formatter: function (param) {
          const count = param.value[2];
          return count >= 1e6
            ? (count / 1e6).toFixed(1) + 'M'
            : count >= 1e3
            ? (count / 1e3).toFixed(0) + 'K'
            : count.toString();
        },
        fontSize: 10,
        color: '#000'
      },
      data: epistemic_heatmap.map(d => [
        d.knowledge_level_label,
        d.agent_type_label,
        d.edge_count,
        d.average_score
      ]),
      emphasis: {
        itemStyle: {
          borderColor: '#333',
          borderWidth: 1
        }
      }
    }]
  }}
/>
</div>

## Disease list nodes connections

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
    swapXY=true
    title="Categories connected to disease list node on average"
/>

### Disease nodes neighbours

```sql disease_list_neighbour_counts
select 
  * 
from 
  bq.disease_list_neighbour_counts
```

<ECharts
    style={{ height: '400px' }}
    config={getHistogramEchartsOptions(disease_list_neighbour_counts, "disease", "unique_neighbours", 10)}
/>

<br/>

## Drug list nodes connections

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
    swapXY=true
    title="Categories connected to drug list node on average"
/>

### Drug nodes neighbours

\
```sql drug_list_neighbour_counts
select 
  * 
from 
  bq.drug_list_neighbour_counts
```

<ECharts
    style={{ height: '400px' }}
    config={getHistogramEchartsOptions(drug_list_neighbour_counts, "drug", "unique_neighbours", 50)}
/>
