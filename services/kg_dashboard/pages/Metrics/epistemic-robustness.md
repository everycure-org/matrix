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
SELECT
  SUM(edge_count) AS included_edges,
  ROUND(SUM(kl_score * edge_count) / SUM(edge_count), 4) AS average_knowledge_level
FROM bq.epistemic_scores
```

```sql agent_type_score
SELECT
  SUM(edge_count) AS included_edges,
  ROUND(SUM(at_score * edge_count) / SUM(edge_count), 4) AS average_agent_type
FROM bq.epistemic_scores
```

```sql knowledge_level_by_source
SELECT
  knowledge_level_label AS knowledge_level,
  upstream_data_source,
  SUM(edge_count) AS edge_count
FROM bq.epistemic_scores
GROUP BY knowledge_level_label, upstream_data_source
```

```sql agent_type_by_source
SELECT
  agent_type_label AS agent_type,
  upstream_data_source,
  SUM(edge_count) AS edge_count
FROM bq.epistemic_scores
GROUP BY agent_type_label, upstream_data_source
```
<!-- Explanatory header -->
<div class="text-left text-md max-w-3xl mx-auto mb-6">
  This page provides a deeper look at the provenance quality of the knowledge graph, breaking it down into detailed 
  distributions of 
  <a class="underline text-blue-600" href="https://biolink.github.io/biolink-model/docs/KnowledgeLevel/" target="_blank">Knowledge Level</a> 
  and 
  <a class="underline text-blue-600" href="https://biolink.github.io/biolink-model/docs/AgentType/" target="_blank">Agent Type</a> 
  scores are distributed across all edges. 
</div>
<div class="text-left text-md max-w-3xl mx-auto mb-6">
  Together, these metrics offer a way to assess the overall strength and reliability 
  of the knowledge graph, revealing which regions are supported by direct evidence and expert curation, and which rely on 
  more speculative or indirect information. By exploring these patterns, we gain deeper insight into where the graph is 
  most robust and where caution may be warranted in interpretation.
</div>

<div class="text-left text-lg font-semibold mt-6 mb-2 max-w-3xl mx-auto">
  Knowledge Level
  <div class="text-sm font-normal mt-1 leading-snug">
    Indicates how strong or certain a statement is—ranging from direct assertions and logical entailments 
    to statistical associations and predictions.
  </div>
  <div class="text-sm font-normal mt-1 leading-snug">
    Higher averages reflect greater epistemic confidence; lower values indicate more speculative knowledge.
  </div>
</div>

<Details title="Understand more about Knowledge Level values">
  <div class="max-w-2xl mx-auto text-sm leading-tight text-gray-600 mt-.5 pl-0 ml-0">
    <strong>Knowledge Assertion:</strong> Direct statements such as 'X inhibits Y'.<br/>
    <strong>Logical Entailment:</strong> Inferred relationships (e.g., subclass_of).<br/>
    <strong>Statistical Association:</strong> Based on statistical correlation or co-occurrence.<br/>
    <strong>Observation:</strong> Derived from experimental data.<br/>
    <strong>Prediction:</strong> Predicted by computational models.<br/>
    <strong>Not Provided:</strong> Not specified.
  </div>
</Details>

<div class="mb-6"></div>

<Grid col=2 class="max-w-4xl mx-auto">
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={knowledge_level_score} column="average_knowledge_level" fmt="num2" />
    </span><br/>
    Average Knowledge Level
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={knowledge_level_score} column="included_edges" fmt="num2m" />
    </span><br/>
    edges used in calculation
  </div>
</Grid>

<ECharts 
  style={{ height: '700px' }}
  config={{
    title: { text: 'Knowledge Level by Upstream Data Source', left: 'center' },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
    },
    legend: { top: 20 },
    grid: { top: 50, left: '3%', right: '4%', bottom: '15%', containLabel: true },
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
        "Knowledge Assertion", "Logical Entailment", "Statistical Association",
        "Observation", "Prediction", "Not Provided", "null"
      ],
      axisLabel: { rotate: 30, fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: v => (v / 1000).toLocaleString() + 'k'
      }
    },
    series: Object.entries(groupBy(knowledge_level_by_source, 'upstream_data_source')).map(([source, values]) => ({
      name: source,
      type: 'bar',
      stack: 'total',
      emphasis: { focus: 'series' },
      data: [
        "Knowledge Assertion", "Logical Entailment", "Statistical Association",
        "Observation", "Prediction", "Not Provided", "null"
      ].map(k => (values.find(v => v.knowledge_level === k) || {}).edge_count || 0)
    }))
  }}
/>

<div class="text-left text-lg font-semibold mt-10 mb-2 max-w-3xl mx-auto">
  Agent Type
  <div class="text-sm font-normal mt-1 leading-snug">
    Describes how the knowledge was generated — from direct human curation to automated text mining or computational models.
  </div>
  <div class="text-sm font-normal mt-1 leading-snug">
    Higher averages indicate more human involvement; lower values reflect more automated or speculative sources.
  </div>
</div>

<Details title="Understand more about Agent Type values">
  <div class="max-w-2xl mx-auto text-sm leading-tight text-gray-600 mt-.5 pl-0 ml-0">
    <strong>Manual Agent:</strong> Manually curated by domain experts.<br/>
    <strong>Manual Validation of Automated Agent:</strong> Machine generated then manually validated.<br/>
    <strong>Automated Agent:</strong> Automatically generated by pipelines.<br/>
    <strong>Data Analysis Pipeline:</strong> Derived via data analysis workflows.<br/>
    <strong>Computational Model:</strong> Predicted by computational simulations.<br/>
    <strong>Text-Mining Agent:</strong> Extracted from literature via NLP.<br/>
    <strong>Image Processing Agent:</strong> Derived from image analysis.<br/>
    <strong>Not Provided:</strong> Agent type not specified.
  </div>
</Details>

<div class="mb-6"></div>

<Grid col=2 class="max-w-4xl mx-auto">
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={agent_type_score} column="average_agent_type" fmt="num2" />
    </span><br/>
    Average Agent Type
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={agent_type_score} column="included_edges" fmt="num2m" />
    </span><br/>
    edges used in calculation
  </div>
</Grid>

<ECharts 
  style={{ height: '750px' }}
  config={{
    title: { text: 'Agent Type by Upstream Data Source', left: 'center' },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }},
    legend: { top: 20 },
    grid: { top: 50, left: '3%', right: '4%', bottom: '15%', containLabel: true },
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
        "Manual Agent", "Manual Validation\nof Automated Agent", "Data Analysis Pipeline",
        "Automated Agent", "Computational Model", "Text-Mining Agent",
        "Image Processing\nAgent", "Not Provided", "null"
      ],
      axisLabel: { rotate: 30, fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: v => (v / 1000).toLocaleString() + 'k'
      }
    },
    series: Object.entries(groupBy(agent_type_by_source, 'upstream_data_source')).map(([source, values]) => ({
      name: source,
      type: 'bar',
      stack: 'total',
      emphasis: { focus: 'series' },
      data: [
        "Manual Agent", "Manual Validation\nof Automated Agent", "Data Analysis Pipeline",
        "Automated Agent", "Computational Model", "Text-Mining Agent",
        "Image Processing\nAgent", "Not Provided", "null"
      ].map(k => (values.find(v => v.agent_type === k) || {}).edge_count || 0)
    }))
  }}
/>
