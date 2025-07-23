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

<DataTable data={source_kg_summary} link=link >
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

<SankeyDiagram 
  data={source_kg_flow} 
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  chartAreaHeight={600}
  valueFmt='0,0'
  title='Source KG Contributions to Unified Graph'
  subtitle='Flow of edges from individual source KGs into the unified knowledge graph'
/>

