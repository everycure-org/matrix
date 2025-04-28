---
title: Summary
---
<script>
  function groupBy(arr, key) {
    return arr.reduce((acc, item) => {
      const group = item[key];
      acc[group] = acc[group] || [];
      acc[group].push(item);
      return acc;
    }, {});
  }
</script>

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

```sql knowledge_level_score
SELECT * FROM bq.knowledge_level_score
```

```sql agent_type_score
SELECT * FROM bq.agent_type_score
```

```sql knowledge_level_by_source
SELECT * FROM bq.knowledge_level_distribution
```

```sql agent_type_by_source
SELECT * FROM bq.agent_type_distribution
```

```sql epistemic_score
SELECT * FROM bq.epistemic_score
```

```sql epistemic_heatmap
SELECT * FROM bq.epistemic_heatmap
```

<div class="text-center text-lg font-semibold mt-6 mb-2">
    Epistemic Score
    <div class="text-sm font-normal mt-1">
        The Epistemic Score summarizes provenance quality across the graph by averaging values assigned to each edge's
        knowledge level and agent type.
    </div>
    <div class="text-sm font-normal mt-1">
        A more positive average reflects stronger overall provenance, while a more negative average indicates weaker 
        or more speculative knowledge across the graph.
    </div>
</div>

<!-- Spacer -->
<div class="mb-6"></div>

<!-- Metric row: Epistemic Score -->
<Grid col=2>
  <div class="text-center text-lg">
    <p>
      <span class="font-semibold text-2xl">
        <Value data={epistemic_score} column="average_epistemic_score" fmt="num2" />
      </span><br/>
      Epistemic Score
    </p>
  </div>
  <div class="text-center text-lg">
    <p>
      <span class="font-semibold text-2xl">
        <Value data={epistemic_score} column="included_edges" fmt="num2m" />
      </span><br/>
      edges used in calculation
    </p>
  </div>
</Grid>

<!-- Spacer -->
<div class="mb-6"></div>

<!-- heatmap -->
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
        "Prediction", 
        "Observation", 
        "Not Provided", 
        "Statistical Association", 
        "Logical Entailment", 
        "Knowledge Assertion",
      ],
      axisLabel: {
        rotate: 30,
        fontSize: 10
      }
    },
    yAxis: {
      type: 'category',
      data: [
        "Text Mining Agent",
        "Image Processing\nAgent",
        "Not Provided", 
        "Computational Model",
        "Data Analysis Pipeline",
        "Automated Agent",
        "Manual Validation of\nAutomated Agent",
        "Manual Agent"
      ],
      axisLabel: {
        fontSize: 10
      }
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

<!-- Spacer -->
<div class="mb-6"></div>

<div class="text-center text-lg font-semibold mt-6 mb-2">
    Knowledge Level
    <div class="text-sm font-normal mt-1">
        Indicates how strong or certain a statement is—ranging from direct assertions and logical entailments to 
        predictions and statistical associations.
    </div>
    <div class="text-sm font-normal mt-1">
      A more positive average reflects greater epistemic confidence, while a more negative average indicates weaker 
      or more speculative knowledge.
    </div>
</div>

<!-- Spacer -->
<div class="mb-6"></div>

<!-- First row: metrics side-by-side -->
<Grid col=2>
  <div class="text-center text-lg">
    <p>
      <span class="font-semibold text-2xl">
        <Value data={knowledge_level_score} column="average_knowledge_level" fmt="num2" />
      </span><br/>
      Average Knowledge Level
    </p>
  </div>
  <div class="text-center text-lg">
    <p>
      <span class="font-semibold text-2xl">
        <Value data={knowledge_level_score} column="included_edges" fmt="num2m" />
      </span><br/>
      edges used in calculation
    </p>
  </div>
</Grid>

<br/>

<!-- Second row: full-width bar chart -->
<ECharts 
  style={{ height: '700px' }}
  config={{
    title: {
      text: 'Knowledge Level by Upstream Data Source',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    legend: {
      top: 20
    },
    grid: {
      top: 50,
      left: '3%',
      right: '4%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: [
        "Knowledge Assertion",
        "Logical Entailment",
        "Statistical Association",
        "Observation",
        "Prediction",
        "Not Provided",
        "null"
      ],
      axisLabel: {
        rotate: 30,
        fontSize: 10
      }
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: function (value) {
          return (value / 1000).toLocaleString() + 'k';
    }
  }
    },
    series: Object.entries(groupBy(knowledge_level_by_source, 'upstream_data_source')).map(([source, values]) => ({
      name: source,
      type: 'bar',
      stack: 'total',
      emphasis: {
        focus: 'series'
      },
      data: [
        "Knowledge Assertion",
        "Logical Entailment",
        "Statistical Association",
        "Observation",
        "Prediction",
        "Not Provided",
        "null"
      ].map(k => {
        const entry = values.find(v => v.knowledge_level === k);
        return entry ? entry.edge_count : 0;
      })
    }))
  }}
/>


<div class="text-center text-lg font-semibold mt-6 mb-2">
    Agent Type
    <div class="text-sm font-normal mt-1">
        Agent Type describes the origin of an edge in terms of how the knowledge was generated—ranging 
        from direct human assertions to automated text mining.
    </div>
    <div class="text-sm font-normal mt-1">
        A more positive average reflects greater human involvement, while a more negative average indicates greater 
        reliance on automated or speculative sources.
    </div>
</div>

<!-- Spacer -->
<div class="mb-6"></div>


<!-- First row: metrics side-by-side -->
<Grid col=2>
  <div class="text-center text-lg">
    <p>
      <span class="font-semibold text-2xl">
        <Value data={agent_type_score} column="average_agent_type" fmt="num2" />
      </span><br/>
      Average Agent Type
    </p>
  </div>
  <div class="text-center text-lg">
    <p>
      <span class="font-semibold text-2xl">
        <Value data={agent_type_score} column="included_edges" fmt="num2m" />
      </span><br/>
      edges used in calculation
    </p>
  </div>
</Grid>

<br/>

<!-- Second row: full-width bar chart -->
<ECharts 
  style={{ height: '750px' }}
  config={{
    title: {
      text: 'Agent Type by Upstream Data Source',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    legend: {
      top: 20
    },
    grid: {
      top: 50,
      left: '3%',
      right: '4%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: [
        "Manual Agent",
        "Manual Validation\nof Automated Agent",
        "Data Analysis Pipeline",
        "Automated Agent",
        "Computational Model",
        "Text-Mining Agent",
        "Image Processing\nAgent",
        "Not Provided",
        "null"
      ],
      axisLabel: {
        rotate: 30,
        fontSize: 10
      }
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: function (value) {
          return (value / 1000).toLocaleString() + 'k';
    }
  }
    },
    series: Object.entries(groupBy(agent_type_by_source, 'upstream_data_source')).map(([source, values]) => ({
      name: source,
      type: 'bar',
      stack: 'total',
      emphasis: {
        focus: 'series'
      },
      data: [
        "Manual Agent",
        "Manual Validation\nof Automated Agent",
        "Data Analysis Pipeline",
        "Automated Agent",
        "Computational Model",
        "Text-Mining Agent",
        "Image Processing\nAgent",
        "Not Provided",
        "null"
      ].map(k => {
        const entry = values.find(v => v.agent_type === k);
        return entry ? entry.edge_count : 0;
      })
    }))
  }}
/>


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