---
title: Knowledge Sources
---
<p>
This page presents the primary knowledge sources (KSs) ingested into the Matrix graph, offering an overview of the individual 
sources that compose the graph, along with key metrics and detailed insights for each source.
</p>

``` sql distinct_primary_knowledge_source
SELECT DISTINCT 
    primary_knowledge_source as value,
    concat(primary_knowledge_source, ' (', sum(count), ' connections)') as label
FROM bq.merged_kg_edges
WHERE primary_knowledge_source IS NOT NULL
GROUP BY primary_knowledge_source
ORDER BY primary_knowledge_source
```

```sql distinct_upstream_knowledge_source
SELECT 
  CASE 
    WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
    THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
    ELSE TRIM(upstream_source)
  END AS value,
  concat(
    CASE 
      WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
      THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
      ELSE TRIM(upstream_source)
    END,
    ' (', sum(count), ' connections)'
  ) AS label
FROM bq.merged_kg_edges
CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) AS t(upstream_source)
WHERE TRIM(upstream_source) IS NOT NULL
  AND TRIM(upstream_source) != ''
GROUP BY 1
ORDER BY label
```

<script>
  // Create depth overrides for proper Sankey layout
  let depthOverrides = {}
  
  if (distinct_primary_knowledge_source && Array.isArray(distinct_primary_knowledge_source)) {    
    distinct_primary_knowledge_source.forEach(pks => {
      depthOverrides[pks.value] = 0;
    });    
  }

  if (distinct_upstream_knowledge_source && Array.isArray(distinct_upstream_knowledge_source)) {
    distinct_upstream_knowledge_source.forEach(uks => {
      depthOverrides[uks.value] = 1;
    });
  }

  // Unified KG is always at depth 2
  depthOverrides['Unified KG'] = 2;
  
  // Add common cleaned upstream source names to ensure proper depth
  const commonUpstreamSources = ['ec_medical', 'rtxkg2', 'robokop'];
  commonUpstreamSources.forEach(source => {
    depthOverrides[source] = 1;
  });
</script>

```sql knowledge_source_sankey
-- First level: Primary Knowledge Source to Upstream Data Source
select 
    primary_knowledge_source as source, 
    CASE 
      WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
      THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
      ELSE TRIM(upstream_source)
    END as target, 
    sum(count) as count
from bq.merged_kg_edges
cross join UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
WHERE primary_knowledge_source IN ${inputs.selected_primary_sources.value}
  AND CASE 
    WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
    THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
    ELSE TRIM(upstream_source)
  END IN ${inputs.selected_upstream_sources.value}
  AND TRIM(upstream_source) IS NOT NULL
  AND TRIM(upstream_source) != ''
group by all

UNION ALL

-- Second level: Upstream Data Source to Unified KG (filtered)
select 
    CASE 
      WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
      THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
      ELSE TRIM(upstream_source)
    END as source,
    'Unified KG' as target,
    sum(count) as count
from bq.merged_kg_edges
cross join UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
WHERE CASE 
  WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
  THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
  ELSE TRIM(upstream_source)
END IN ${inputs.selected_upstream_sources.value}
  AND primary_knowledge_source IN ${inputs.selected_primary_sources.value}
  AND TRIM(upstream_source) IS NOT NULL
  AND TRIM(upstream_source) != ''
group by source

order by count desc
```

## Filter Knowledge Sources
Use the filters below to refine your view of associations in the Matrix Knowledge Graph. You can limit the visualization to specific primary knowledge sources or upstream knowledge sources.

<div>
<Dropdown
  data={distinct_primary_knowledge_source}
  name=selected_primary_sources
  value=value
  label=label
  title="Filter Primary KS"
  multiple=true
  selectAllByDefault=true
  description="Filter knowledge graph by primary knowledge sources"
/>
</div>

<div>
<Dropdown
  data={distinct_upstream_knowledge_source}
  name=selected_upstream_sources
  value=value
  label=label
  title="Filter Upstream KS"
  multiple=true
  selectAllByDefault=true
  description="Filter knowledge graph by upstream sources"
/>
</div>

## Knowledge Source Flow

<SankeyDiagram 
  data={knowledge_source_sankey} 
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
    chartAreaHeight={1200}
  valueFmt='0,0'
  depthOverride={depthOverrides}
/>

## Knowledge Source Details

```sql knowledge_source_table
SELECT 
  primary_knowledge_source.source,
  catalog.name as name,
  '/Knowledge Sources/' || primary_knowledge_source.source as link,
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
