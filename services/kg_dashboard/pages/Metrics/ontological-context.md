---
title: Ontological Context
---

```sql ontological_aggregate
WITH total_edges_all_sources AS (
  SELECT
    SUM(count) as total_edges
  FROM bq.merged_kg_edges
)
SELECT
  SUM(m.abox) as abox,
  SUM(m.tbox) as tbox,
  MAX(t.total_edges) - (SUM(m.abox) + SUM(m.tbox)) as undefined,
  MAX(t.total_edges) as total_edges,
  ROUND(100.0 * SUM(m.abox) / NULLIF(MAX(t.total_edges), 0), 1) as abox_percentage,
  ROUND(100.0 * SUM(m.tbox) / NULLIF(MAX(t.total_edges), 0), 1) as tbox_percentage,
  ROUND(100.0 * (MAX(t.total_edges) - (SUM(m.abox) + SUM(m.tbox))) / NULLIF(MAX(t.total_edges), 0), 1) as undefined_percentage
FROM bq.abox_tbox_metric m
CROSS JOIN total_edges_all_sources t
```

```sql ontological_chart
WITH total_edges_all_sources AS (
  SELECT
    SUM(count) as total_edges
  FROM bq.merged_kg_edges
)
SELECT
  'ABox (Instance-level)' as edge_type,
  SUM(m.abox) as count
FROM bq.abox_tbox_metric m
CROSS JOIN total_edges_all_sources t

UNION ALL

SELECT
  'TBox (Concept-level)' as edge_type,
  SUM(m.tbox) as count
FROM bq.abox_tbox_metric m
CROSS JOIN total_edges_all_sources t

UNION ALL

SELECT
  'Undefined' as edge_type,
  MAX(t.total_edges) - (SUM(m.abox) + SUM(m.tbox)) as count
FROM bq.abox_tbox_metric m
CROSS JOIN total_edges_all_sources t
```

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  The TBox-to-ABox balance reflects how much a knowledge graph emphasizes abstract schema versus concrete instances—too much of either can hinder effective learning and reasoning.
</div>

<Details title="About This Metric">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
ABox (Assertional Box) edges represent instance-level relationships between specific entities, while TBox
(Terminological Box) edges represent concept-level relationships between types or classes.
This classification helps understand the semantic nature of the knowledge graph.
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
A high TBox ratio suggests the graph contains mostly general, ontological structure, which may add noise and limit the
discovery of meaningful, specific patterns by machine learning models.
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
Conversely, a low TBox ratio might mean the graph is missing useful schema-level structure that could aid clustering or
reasoning - e.g., similar nodes may not be grouped efficiently in embedding space.
</div>
</Details>

{#if ontological_aggregate.length > 0}
<Grid col=2 class="max-w-4xl mx-auto mb-8 mt-6">
  <div>
    <ECharts config={{
        title: {
            text: 'Classification',
            left: 'center',
            top: 'center',
            textStyle: {
                fontWeight: 'normal'
            }
        },
        color: ['#6287D3', '#D8AB47', '#AAAAAA'],
        tooltip: {
            formatter: function(params) {
                const count = params.data.value.toLocaleString();
                return `${params.name}: ${count} edges (${params.percent}%)`;
            }
        },
        series: [{
            type: 'pie',
            data: ontological_chart.map(row => ({
              value: row.count,
              name: row.edge_type
            })),
            radius: ['40%', '65%']
        }]
    }}/>
  </div>
  <div class="text-center flex flex-col justify-center">
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: #6287D3;">
        <Value data={ontological_aggregate} column="abox" fmt="num0" />
      </span><br/>
      <span class="text-base">ABox Edges</span><br/>
      <span class="text-xs text-gray-600">
        (<Value data={ontological_aggregate} column="abox_percentage" fmt="num1" />%)
      </span>
    </div>
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: #D8AB47;">
        <Value data={ontological_aggregate} column="tbox" fmt="num0" />
      </span><br/>
      <span class="text-base">TBox Edges</span><br/>
      <span class="text-xs text-gray-600">
        (<Value data={ontological_aggregate} column="tbox_percentage" fmt="num1" />%)
      </span>
    </div>
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: #AAAAAA;">
        <Value data={ontological_aggregate} column="undefined" fmt="num0" />
      </span><br/>
      <span class="text-base">Undefined</span><br/>
      <span class="text-xs text-gray-600">
        (<Value data={ontological_aggregate} column="undefined_percentage" fmt="num1" />%)
      </span>
    </div>
  </div>
</Grid>
{:else}
<div class="text-center text-lg text-gray-500 mb-6">
  No ontological context data available.
</div>
{/if}

## Breakdown by Knowledge Source

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  The table below shows the ABox/TBox classification metrics for each knowledge source in the graph.
  Click on a source to see detailed metrics.
</div>

```sql ontological_by_source
WITH total_edges_per_source AS (
  SELECT
    primary_knowledge_source,
    SUM(count) as total_edges
  FROM bq.merged_kg_edges
  GROUP BY primary_knowledge_source
)
SELECT
  m.primary_knowledge_source as source,
  catalog.name as name,
  '/Knowledge Sources/' || m.primary_knowledge_source as link,
  m.abox,
  m.tbox,
  t.total_edges - (m.abox + m.tbox) as undefined,
  t.total_edges,
  ROUND(100.0 * m.abox / NULLIF(t.total_edges, 0), 1) as abox_pct,
  ROUND(100.0 * m.tbox / NULLIF(t.total_edges, 0), 1) as tbox_pct,
  ROUND(100.0 * (t.total_edges - (m.abox + m.tbox)) / NULLIF(t.total_edges, 0), 1) as undefined_pct
FROM bq.abox_tbox_metric m
JOIN total_edges_per_source t ON m.primary_knowledge_source = t.primary_knowledge_source
LEFT JOIN infores.catalog ON catalog.id = m.primary_knowledge_source
ORDER BY t.total_edges DESC
```

<DataTable data={ontological_by_source} link=link search=true>
  <Column id="source" title="Knowledge Source ID" />
  <Column id="name" title="Name" />
  <Column id="abox_pct" title="ABox %" fmt="num1" contentType="colorscale" scaleColor="#6287D3" />
  <Column id="tbox_pct" title="TBox %" fmt="num1" contentType="colorscale" scaleColor="#D8AB47" />
  <Column id="undefined_pct" title="Undefined %" fmt="num1" contentType="colorscale" scaleColor="#AAAAAA" />
  <Column id="abox" title="ABox" fmt="num0" />
  <Column id="tbox" title="TBox" fmt="num0" />
  <Column id="undefined" title="Undefined" fmt="num0" />
  <Column id="total_edges" title="Total Edges" fmt="num0" />
</DataTable>
