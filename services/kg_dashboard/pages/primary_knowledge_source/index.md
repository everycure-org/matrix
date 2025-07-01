---
title: Primary Knowledge Source
---

This page presents the primary knowledge sources ingested into the Matrix graph, offering an overview of the individual sources that compose the graph, along with key metrics and detailed insights for each source.

```sql knowledge_source_provenance
SELECT primary_knowledge_source.*, catalog.name as name 
FROM bq.primary_knowledge_source
LEFT OUTER JOIN infores.catalog on infores.catalog.id = bq.primary_knowledge_source.source
```

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

<DataTable data={knowledge_source_provenance} />
