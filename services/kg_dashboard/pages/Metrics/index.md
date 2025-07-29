---
title: Metrics
category: Metrics
---


## Epistemic Robustness

```sql epistemic_score
SELECT * FROM bq.epistemic_score
```

```sql epistemic_heatmap
SELECT * FROM bq.epistemic_heatmap
```

<div class="text-left text-md mt-6 mb-4">
  Epistemic Robustness summarizes the provenance quality of the knowledge graph, using Knowledge Level and Agent Type  
  from the Biolink Model to assess the strength and reliability of edges. Higher scores indicate stronger evidence and 
  greater manual curation. Explore below for both an overall view and visit 
  <a class="underline text-blue-600" href="./Metrics/epistemic-robustness">Epistemic Robustness</a> 
  for a detailed breakdown.
</div>


  <div class="text-center">
    <span class="font-semibold text-2xl">
      <Value data={epistemic_score} column="average_epistemic_score" fmt="num2" />
    </span><br/>
    Epistemic Score
  </div>

<Grid col=2>
  <div class="text-center">
    <span class="font-semibold text-xl">
      <Value data={epistemic_score} column="included_edges" fmt="num2m" />
    </span><br/>
    edges included
  </div>
  <div class="text-center">
    <span class="font-semibold text-xl">
      <Value data={epistemic_score} column="null_or_not_provided_both" fmt="num2m" />
    </span><br/>
    edges with missing provenance
    <div class="text-xs font-normal mt-.25">
      (Both Knowledge Level and Agent Type are not provided)
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
