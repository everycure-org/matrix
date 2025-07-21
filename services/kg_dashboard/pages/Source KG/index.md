---
title: Source KG
---

<p>
This page presents the source knowledge graphs (KGs) that have been merged into the unified Matrix graph, offering an overview of the individual 
KGs and their contributions to the overall knowledge graph.
</p>

## Source KG Overview

```sql source_kg_summary
SELECT 
  upstream_data_source,
  n_nodes,
  n_edges,
  '/Source KG/' || upstream_data_source as link
FROM bq.upstream_data_sources
WHERE upstream_data_source IS NOT NULL
ORDER BY n_edges DESC
```

<DataTable data={source_kg_summary} link=link search=true>
  <Column id="upstream_data_source" title="Source KG" />
  <Column id="n_nodes" title="Nodes" fmt="num0" />
  <Column id="n_edges" title="Edges" fmt="num0" />
</DataTable>

## Source KG Flow

```sql source_kg_flow
-- Flow from Source KGs to Unified KG
SELECT 
  TRIM(upstream_source) as source,
  'Unified KG' as target,
  SUM(count) as count
FROM bq.merged_kg_edges
CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) AS t(upstream_source)
WHERE TRIM(upstream_source) IS NOT NULL
  AND TRIM(upstream_source) != ''
GROUP BY source
ORDER BY count DESC
```

<script>
  // Create depth overrides for proper Sankey layout
  let depthOverrides = {}
  
  if (source_kg_flow && Array.isArray(source_kg_flow)) {
    source_kg_flow.forEach(flow => {
      depthOverrides[flow.source] = 0;
    });
  }

  // Unified KG is always at depth 1
  depthOverrides['Unified KG'] = 1;
</script>

<SankeyDiagram 
  data={source_kg_flow} 
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  chartAreaHeight={600}
  valueFmt='0,0'
  depthOverride={depthOverrides}
  title='Source KG Contributions to Unified Graph'
  subtitle='Flow of edges from individual source KGs into the unified knowledge graph'
/>

## Source KG Statistics

```sql source_kg_stats
SELECT 
  COUNT(DISTINCT upstream_data_source) as total_source_kgs,
  SUM(n_nodes) as total_nodes,
  SUM(n_edges) as total_edges,
  AVG(n_edges) as avg_edges_per_source,
  MAX(n_edges) as max_edges_per_source,
  MIN(n_edges) as min_edges_per_source
FROM bq.upstream_data_sources
WHERE upstream_data_source IS NOT NULL
```

<Grid col=3 class="max-w-4xl mx-auto mb-6">
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={source_kg_stats} column="total_source_kgs" fmt="num0" />
    </span><br/>
    Source KGs
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={source_kg_stats} column="total_nodes" fmt="num2m" />
    </span><br/>
    Total Nodes
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={source_kg_stats} column="total_edges" fmt="num2m" />
    </span><br/>
    Total Edges
  </div>
</Grid>