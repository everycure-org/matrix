---
title: Epistemic Robustness
---

<script>
  import { sourceColorMap } from '../../_lib/colors';
  
  function groupBy(arr, key) {
    return arr.reduce((acc, item) => {
      const group = item[key];
      acc[group] = acc[group] || [];
      acc[group].push(item);
      return acc;
    }, {});
  }
  
  // Function to get colors for ECharts series
  function getEChartsColors(dataSourceGroups) {
    return Object.keys(dataSourceGroups).map(source => 
      sourceColorMap[source] || "#6b7280"
    );
  }
</script>


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
## Graph Relevancy Metrics


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
    color: getEChartsColors(groupBy(knowledge_level_by_source, 'upstream_data_source')),
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
    color: getEChartsColors(groupBy(agent_type_by_source, 'upstream_data_source')),
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
