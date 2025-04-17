---
title: RTX-KG2 Normalization
---

<script>
  // Build funnel-style data with dropped counts and tooltips
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
          (dropped !== null
            ? `<br/>Dropped from previous: ${dropped.toLocaleString()} (−${droppedPct}%)`
            : '')
      };
    });
  }

  // Compute the maximum value in a dataset (fallback to 1)
  function getMaxValue(data) {
    if (!data || !data.length) return 1;
    return Math.max(...data.map(d => d.value), 1);
  }
</script>

```sql rtx_kg2_node_summary
SELECT
name,
value
FROM bq.rtx_kg2_node_summary
```

```sql rtx_kg2_edge_summary
SELECT
name,
value
FROM bq.rtx_kg2_edge_summary
```


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


<Grid col={2}>
  <ECharts
    style={{ height: '500px' }}
    config={{
      title: { text: 'RTX‑KG2 Node Normalization', left: 'center' },
      tooltip: { trigger: 'item', formatter: p => p.data.tooltipText },
      legend: {
        show: true,
        orient: 'horizontal',
        type: 'scroll',
        top: 22
      },
      series: [
        {
          type: 'funnel',
          name: 'Value',
          left: '10%',
          top: 53,
          bottom: 10,
          width: '80%',
          min: 0,
          max: getMaxValue(rtx_kg2_node_summary),
          minSize: '30%',
          maxSize: '90%',
          gap: 2,
          sort: 'none',
          funnelAlign: 'center',
          label: { show: true, position: 'inside' },
          labelLine: { length: 10, lineStyle: { width: 1, type: 'solid' } },
          emphasis: { focus: 'series' },
          labelLayout: { hideOverlap: true },
          data: buildFunnelData(rtx_kg2_node_summary)
        }
      ]
    }}
  />

  <ECharts
    style={{ height: '500px' }}
    config={{
      title: { text: 'RTX‑KG2 Edge Normalization', left: 'center' },
      tooltip: { trigger: 'item', formatter: p => p.data.tooltipText },
      legend: {
        show: true,
        orient: 'horizontal',
        type: 'scroll',
        top: 22
      },
      series: [
        {
          type: 'funnel',
          name: 'Value',
          left: '10%',
          top: 53,
          bottom: 10,
          width: '80%',
          min: 0,
          max: getMaxValue(rtx_kg2_edge_summary),
          minSize: '30%',
          maxSize: '90%',
          gap: 2,
          sort: 'none',
          funnelAlign: 'center',
          label: { show: true, position: 'inside' },
          labelLine: { length: 10, lineStyle: { width: 1, type: 'solid' } },
          emphasis: { focus: 'series' },
          labelLayout: { hideOverlap: true },
          data: buildFunnelData(rtx_kg2_edge_summary)
        }
      ]
    }}
  />
</Grid>