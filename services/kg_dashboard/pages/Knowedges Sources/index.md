---
title: Knowledge Sources
---

This page presents the primary knowledge sources ingested into the Matrix graph, offering an overview of the individual sources that compose the graph, along with key metrics and detailed insights for each source.

```sql knowledge_source_sankey
-- First level: Primary Knowledge Source to Upstream Data Source
select 
    primary_knowledge_source as source, 
    TRIM(upstream_source) as target, 
    sum(count) as count
from bq.merged_kg_edges
cross join UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
group by all

UNION ALL

-- Second level: Upstream Data Source to Unified KG
select 
    TRIM(upstream_source) as source,
    'Unified KG' as target,
    sum(count) as count
from bq.merged_kg_edges
cross join UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
group by source

order by count desc
```
## Knowledge Source Flow

<SankeyDiagram 
  data={knowledge_source_sankey} 
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  title='Knowledge Source to Unified KG Flow'
  subtitle='Flow showing how primary knowledge sources connect to their upstream data sources'
  chartAreaHeight={1200}
  valueFmt='0,0'
/>

## Knowledge Source Details

```sql knowledge_source_table
SELECT 
  primary_knowledge_source.source,
  catalog.name as name,
  '/Knowedges Sources/' || primary_knowledge_source.source as link,
  COALESCE(edge_counts.total_edges, 0) as n_edges
FROM (
  SELECT DISTINCT source 
  FROM bq.primary_knowledge_source
) primary_knowledge_source
JOIN infores.catalog on infores.catalog.id = primary_knowledge_source.source
LEFT JOIN (
  SELECT 
    primary_knowledge_source,
    SUM(count) as total_edges
  FROM bq.merged_kg_edges
  GROUP BY primary_knowledge_source
) edge_counts ON edge_counts.primary_knowledge_source = primary_knowledge_source.source
ORDER BY n_edges DESC
```

<DataTable data={knowledge_source_table} link=link search=true>
  <Column id="source" title="Knowledge Source ID" />
  <Column id="name" title="Name" />
  <Column id="n_edges" title="Edges" fmt="num0" />
</DataTable>
