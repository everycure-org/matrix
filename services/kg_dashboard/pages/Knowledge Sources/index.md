---
title: Knowledge Sources
---

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

  // Add "Other" category to depth overrides
  depthOverrides['Other (Small Sources)'] = 0;

 const smallSourceThreshold = 50000
</script>

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
  <Column id="n_edges" title="Edges" contentType="bar" barColor="#93c5fd" backgroundColor="#e5e7eb" fmt="num0" />
</DataTable>

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

```sql knowledge_source_sankey
-- Wrap everything in a subquery so we can ORDER BY the source column
SELECT * FROM (
  -- Pre-calculate source totals for consistent grouping
  WITH source_totals AS (
    SELECT 
      primary_knowledge_source,
      SUM(count) as total_count
    FROM bq.merged_kg_edges
    WHERE primary_knowledge_source IN ${inputs.selected_primary_sources.value}
    GROUP BY primary_knowledge_source
  )

  -- First level: Primary Knowledge Source to Upstream Data Source
  SELECT 
      CASE 
        WHEN '${inputs.view_mode}' = 'detailed' THEN mke.primary_knowledge_source
        WHEN st.total_count < ${smallSourceThreshold} THEN 'Other (Small Sources)'
        ELSE mke.primary_knowledge_source 
      END as source, 
      CASE 
        WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
        THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
        ELSE TRIM(upstream_source)
      END as target, 
      SUM(mke.count) as count
  FROM bq.merged_kg_edges mke
  JOIN source_totals st ON st.primary_knowledge_source = mke.primary_knowledge_source
  CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
  WHERE mke.primary_knowledge_source IN ${inputs.selected_primary_sources.value}
    AND CASE 
      WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
      THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
      ELSE TRIM(upstream_source)
    END IN ${inputs.selected_upstream_sources.value}
    AND TRIM(upstream_source) IS NOT NULL
    AND TRIM(upstream_source) != ''
  GROUP BY 1, 2

  UNION ALL

  -- Second level: Upstream Data Source to Unified KG
  SELECT 
      CASE 
        WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
        THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
        ELSE TRIM(upstream_source)
      END as source,
      'Unified KG' as target,
      SUM(count) as count
  FROM bq.merged_kg_edges
  CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
  WHERE CASE 
    WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
    THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
    ELSE TRIM(upstream_source)
  END IN ${inputs.selected_upstream_sources.value}
    AND primary_knowledge_source IN ${inputs.selected_primary_sources.value}
    AND TRIM(upstream_source) IS NOT NULL
    AND TRIM(upstream_source) != ''
  GROUP BY source
) 
ORDER BY 
  CASE WHEN source = 'Other (Small Sources)' THEN 1 ELSE 0 END,
  count DESC
```

## Filter Knowledge Sources

Use the filters below to refine your view of associations in the Matrix Knowledge Graph. You can limit the visualization to specific primary knowledge sources or upstream knowledge sources.

```sql view_options
SELECT 'simplified' as value, 'Simplified View' as label
UNION ALL SELECT 'detailed', 'Detailed View'
```

### View Options

### Source Filters
<div style="display: flex; gap: 20px;">
  <div style="flex: 1;">
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
  <div style="flex: 1;">
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
</div>

<ButtonGroup
name=view_mode
data={view_options}
value=value
label=label
defaultValue='simplified'
description="Choose between simplified view (groups small sources) or detailed view (shows all sources individually)"
/>
{#if inputs.view_mode === 'detailed'}
  <p style="color: #6b7280; font-size: 14px; margin-bottom: 12px; font-style: italic;">
    Showing all knowledge sources individually
  </p>
{:else}
  <p style="color: #6b7280; font-size: 14px; margin-bottom: 12px; font-style: italic;">
    Sources with fewer than {smallSourceThreshold.toLocaleString()}  edges are grouped as "Other (Small Sources)"
  </p>
{/if}

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