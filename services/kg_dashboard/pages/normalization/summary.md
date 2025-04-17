---
title: Normalization Summary
---

<script>
  function buildFunnelData(data) {
    const stageOrder = ['Ingested', 'Transformed', 'Normalized'];
    const stages = stageOrder.map(stage => data.find(d => d.name === stage));
    const total = stages[0]?.value || 1;

    return stages.map((stage, i) => {
      const prev = i > 0 ? stages[i - 1] : null;
      const percentOfIngested = ((stage.value / total) * 100).toFixed(1);
      const dropped = prev ? prev.value - stage.value : null;
      const droppedPct = dropped !== null ? ((dropped / total) * 100).toFixed(1) : null;

      return {
        name: stage.name,
        value: stage.value,
        tooltipText:
          `${stage.value.toLocaleString()}` +
          `<br/>Percent of Ingested: ${percentOfIngested}%` +
          (dropped !== null ? `<br/>Dropped from previous: ${dropped.toLocaleString()} (−${droppedPct}%)` : '')
      };
    });
  }
</script>

```sql robokop_node_summary
SELECT * FROM bq.robokop_node_summary
```

```sql robokop_node_max_value
SELECT MAX(value) AS max_value FROM bq.robokop_node_summary
```

```sql robokop_edge_summary
SELECT * FROM bq.robokop_edge_summary
```

```sql robokop_edge_max_value
SELECT MAX(value) AS max_value FROM bq.robokop_edge_summary
```

```sql rtx_kg2_node_summary
SELECT * FROM bq.rtx_kg2_node_summary
```

```sql rtx_kg2_node_max_value
SELECT MAX(value) AS max_value FROM bq.rtx_kg2_node_summary
```

```sql rtx_kg2_edge_summary
SELECT * FROM bq.rtx_kg2_edge_summary
```

```sql rtx_kg2_edge_max_value
SELECT MAX(value) AS max_value FROM bq.rtx_kg2_edge_summary
```


<Grid col=2>
  <div>
    <div class="text-lg font-semibold mb-2">ROBOKOP Node Summary</div>
    <DataTable data={robokop_node_summary} />
  </div>

  <div>
    <div class="text-lg font-semibold mb-2">ROBOKOP Edge Summary</div>
    <DataTable data={robokop_edge_summary} />
  </div>
</Grid>

<Grid col=2>
  <ECharts
    style={{ height: '500px' }}
    config={{
      title: {
        text: 'ROBOKOP Node Normalization',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params) {
          return params.data.tooltipText;
        }
      },
      legend: {
        show: 'true',
        orient: 'horizontal',
        type: 'scroll',
        top: 22,
        padding: [
           0,
           0,
           0,
           0
        ]
     },
      series: [{
        type: 'funnel',
        name: 'Value',
        left: '10%',
        top: 53,
        bottom: 10,
        width: '80%',
        min: 0,
        max: robokop_node_max_value[0]?.max_value || 1,
        minSize: '30%',
        maxSize: '90%',
        gap: 2,
        sort: 'none',
        funnelAlign: 'center',
        label: {
          show: true,
          position: 'inside'
        },
        labelLine: {
          length: 10,
          lineStyle: {
            width: 1,
            type: 'solid'
          }
        },
        emphasis: {
          focus: 'series'
        },
        labelLayout: {
          hideOverlap: true
        },
        data: buildFunnelData(robokop_node_summary)
      }]
    }}
  />
<ECharts
    style={{ height: '500px' }}
    config={{
      title: {
        text: 'ROBOKOP Edge Normalization',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params) {
          return params.data.tooltipText;
        }
      },
      legend: {
        show: 'true',
        orient: 'horizontal',
        type: 'scroll',
        top: 22,
        padding: [
           0,
           0,
           0,
           0
        ]
     },
      series: [{
        type: 'funnel',
        name: 'Value',
        left: '10%',
        top: 53,
        bottom: 10,
        width: '80%',
        min: 0,
        max: robokop_edge_max_value[0]?.max_value || 1,
        minSize: '30%',
        maxSize: '90%',
        gap: 2,
        sort: 'none',
        funnelAlign: 'center',
        label: {
          show: true,
          position: 'inside'
        },
        labelLine: {
          length: 10,
          lineStyle: {
            width: 1,
            type: 'solid'
          }
        },
        emphasis: {
          focus: 'series'
        },
        labelLayout: {
          hideOverlap: true
        },
        data: buildFunnelData(robokop_edge_summary)
      }]
    }}
  />
</Grid>

<Grid col=2>
  <div>
    <div class="text-lg font-semibold mb-2">RTX-KG2 Node Summary</div>
    <DataTable data={rtx_kg2_node_summary} />
  </div>

  <div>
    <div class="text-lg font-semibold mb-2">RTX-KG2 Edge Summary</div>
    <DataTable data={rtx_kg2_edge_summary} />
  </div>
</Grid>


<Grid col=2>
  <ECharts
    style={{ height: '500px' }}
    config={{
      title: {
        text: 'RTX-KG2 Node Normalization',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params) {
          return params.data.tooltipText;
        }
      },
      legend: {
        show: 'true',
        orient: 'horizontal',
        type: 'scroll',
        top: 22,
        padding: [
           0,
           0,
           0,
           0
        ]
     },
      series: [{
        type: 'funnel',
        name: 'Value',
        left: '10%',
        top: 53,
        bottom: 10,
        width: '80%',
        min: 0,
        max: rtx_kg2_node_max_value[0]?.max_value || 1,
        minSize: '30%',
        maxSize: '90%',
        gap: 2,
        sort: 'none',
        funnelAlign: 'center',
        label: {
          show: true,
          position: 'inside'
        },
        labelLine: {
          length: 10,
          lineStyle: {
            width: 1,
            type: 'solid'
          }
        },
        emphasis: {
          focus: 'series'
        },
        labelLayout: {
          hideOverlap: true
        },
        data: buildFunnelData(rtx_kg2_node_summary)
      }]
    }}
  />
<ECharts
    style={{ height: '500px' }}
    config={{
      title: {
        text: 'RTX-KG2 Edge Normalization',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params) {
          return params.data.tooltipText;
        }
      },
      legend: {
        show: 'true',
        orient: 'horizontal',
        type: 'scroll',
        top: 22,
        padding: [
           0,
           0,
           0,
           0
        ]
     },
      series: [{
        type: 'funnel',
        name: 'Value',
        left: '10%',
        top: 53,
        bottom: 10,
        width: '80%',
        min: 0,
        max: rtx_kg2_edge_max_value[0]?.max_value || 1,
        minSize: '30%',
        maxSize: '90%',
        gap: 2,
        sort: 'none',
        funnelAlign: 'center',
        label: {
          show: true,
          position: 'inside'
        },
        labelLine: {
          length: 10,
          lineStyle: {
            width: 1,
            type: 'solid'
          }
        },
        emphasis: {
          focus: 'series'
        },
        labelLayout: {
          hideOverlap: true
        },
        data: buildFunnelData(rtx_kg2_edge_summary)
      }]
    }}
  />
</Grid>
