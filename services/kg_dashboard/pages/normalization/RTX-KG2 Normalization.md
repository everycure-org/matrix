---
title: RTX-KG2 Normalization
---

<script>
  // Build funnel-style data with dropped counts and tooltips
  function buildFunnelData(data) {
    if (!data || !Array.isArray(data) || data.length === 0) {
      // Return dummy data if no data is available
      return [
        { name: 'Ingested', value: 0, tooltipText: 'No data available' },
        { name: 'Transformed', value: 0, tooltipText: 'No data available' },
        { name: 'Normalized', value: 0, tooltipText: 'No data available' }
      ];
    }

    const stageOrder = ['Ingested', 'Transformed', 'Normalized'];
    const stages = stageOrder.map(stage => {
      const found = data.find(d => d.name === stage);
      return found || { name: stage, value: 0 };
    });
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
    return Math.max(...data.map(d => d.value || 0), 1);
  }
</script>

```sql rtx_kg2_node_summary
SELECT
name,
value
FROM bq.rtx_kg2_node_summary
ORDER BY sort_order asc
```

```sql rtx_kg2_edge_summary
SELECT
name,
value
FROM bq.rtx_kg2_edge_summary
ORDER BY sort_order asc
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
      tooltip: { 
        trigger: 'item', 
        formatter: function(params) {
          if (!params.data || !params.data.tooltipText) {
            return 'Loading...';
          }
          return params.data.tooltipText;
        }
      },
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
          max: rtx_kg2_node_summary && rtx_kg2_node_summary.length ? 
               Math.max(...rtx_kg2_node_summary.map(d => d.value || 0)) : 1,
          minSize: '30%',
          maxSize: '90%',
          gap: 2,
          sort: 'none',
          funnelAlign: 'center',
          label: { show: true, position: 'inside' },
          labelLine: { length: 10, lineStyle: { width: 1, type: 'solid' } },
          emphasis: { focus: 'series' },
          labelLayout: { hideOverlap: true },
          data: rtx_kg2_node_summary && rtx_kg2_node_summary.length ? 
                buildFunnelData(rtx_kg2_node_summary) : 
                [
                  { name: 'Ingested', value: 0, tooltipText: 'Loading...' },
                  { name: 'Transformed', value: 0, tooltipText: 'Loading...' },
                  { name: 'Normalized', value: 0, tooltipText: 'Loading...' }
                ]
        }
      ]
    }}
  />

  <ECharts
    style={{ height: '500px' }}
    config={{
      title: { text: 'RTX‑KG2 Edge Normalization', left: 'center' },
      tooltip: { 
        trigger: 'item', 
        formatter: function(params) {
          if (!params.data || !params.data.tooltipText) {
            return 'Loading...';
          }
          return params.data.tooltipText;
        }
      },
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
          max: rtx_kg2_edge_summary && rtx_kg2_edge_summary.length ? 
               Math.max(...rtx_kg2_edge_summary.map(d => d.value || 0)) : 1,
          minSize: '30%',
          maxSize: '90%',
          gap: 2,
          sort: 'none',
          funnelAlign: 'center',
          label: { show: true, position: 'inside' },
          labelLine: { length: 10, lineStyle: { width: 1, type: 'solid' } },
          emphasis: { focus: 'series' },
          labelLayout: { hideOverlap: true },
          data: rtx_kg2_edge_summary && rtx_kg2_edge_summary.length ? 
                buildFunnelData(rtx_kg2_edge_summary) : 
                [
                  { name: 'Ingested', value: 0, tooltipText: 'Loading...' },
                  { name: 'Transformed', value: 0, tooltipText: 'Loading...' },
                  { name: 'Normalized', value: 0, tooltipText: 'Loading...' }
                ]
        }
      ]
    }}
  />
</Grid>