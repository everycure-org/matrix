---
title: Knowledge Sources
---

<script>
  import KnowledgeSourceFlowGraph from '../_components/KnowledgeSourceFlowGraph.svelte';
  
  // Configuration
  const TOP_N_PRIMARY_SOURCES = 25;
  const smallSourceThreshold = 50000;
  
  // Legacy depth overrides (keeping for compatibility with other parts)
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
-- Fixed Sankey query to properly handle deduplication between aggregators
-- The core issue: when the same edge (subject-predicate-object) exists in both 
-- RTX-KG2 and ROBOKOP from the same primary source, it should only be counted once
SELECT * FROM (
  -- Step 1: Get the base unified KG data, deduplicating edges by their actual content
  -- This matches the approach used for calculating the 68M total on the home page
  WITH unified_kg_base AS (
    SELECT 
      subject_prefix,
      subject_category,
      predicate,
      object_prefix,
      object_category,
      primary_knowledge_source,
      upstream_data_source,
      -- Sum counts across aggregators for the same edge
      SUM(count) as total_count
    FROM bq.merged_kg_edges
    WHERE primary_knowledge_source IN ${inputs.selected_primary_sources.value}
    GROUP BY 1, 2, 3, 4, 5, 6, 7
  ),
  
  -- Step 2: Pre-calculate source totals for consistent grouping
  source_totals AS (
    SELECT 
      primary_knowledge_source,
      SUM(total_count) as total_count
    FROM unified_kg_base
    GROUP BY primary_knowledge_source
  ),
  
  -- Step 3: Expand upstream sources and clean them
  expanded_upstream AS (
    SELECT 
      primary_knowledge_source,
      subject_prefix,
      subject_category,
      predicate,
      object_prefix,
      object_category,
      CASE 
        WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
        THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
        ELSE TRIM(upstream_source)
      END as clean_upstream_source,
      total_count
    FROM unified_kg_base
    CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
    WHERE CASE 
        WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
        THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
        ELSE TRIM(upstream_source)
      END IN ${inputs.selected_upstream_sources.value}
      AND TRIM(upstream_source) IS NOT NULL
      AND TRIM(upstream_source) != ''
  )

  -- First level: Primary Knowledge Source to Upstream Data Source
  SELECT 
      CASE 
        WHEN '${inputs.view_mode}' = 'detailed' THEN eu.primary_knowledge_source
        WHEN st.total_count < ${smallSourceThreshold} THEN 'Other (Small Sources)'
        ELSE eu.primary_knowledge_source 
      END as source, 
      eu.clean_upstream_source as target,
      SUM(eu.total_count) as count
  FROM expanded_upstream eu
  JOIN source_totals st ON st.primary_knowledge_source = eu.primary_knowledge_source
  GROUP BY 1, 2

  UNION ALL

  -- Second level: Upstream Data Source to Unified KG
  SELECT 
      eu.clean_upstream_source as source,
      'Unified KG' as target,
      SUM(eu.total_count) as count
  FROM expanded_upstream eu
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

The network diagram below shows how knowledge flows from primary sources through aggregator knowledge graphs (RTX-KG2, ROBOKOP) to create our unified knowledge graph.

```sql network_data
-- Get aggregator-level data for network visualization
WITH base_data AS (
  SELECT 
    primary_knowledge_source,
    CASE 
      WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
      THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
      ELSE TRIM(upstream_source)
    END as clean_upstream_source,
    SUM(count) as edge_count
  FROM bq.merged_kg_edges
  CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
  WHERE primary_knowledge_source IN ${inputs.selected_primary_sources.value}
    AND CASE 
      WHEN TRIM(upstream_source) LIKE '%''unnest'':%'
      THEN TRIM(REPLACE(REPLACE(REPLACE(REPLACE(TRIM(upstream_source), '{''unnest'': ''', ''), '''}', ''), '{''unnest'': ', ''), '}', ''))
      ELSE TRIM(upstream_source)
    END IN ${inputs.selected_upstream_sources.value}
    AND TRIM(upstream_source) IS NOT NULL
    AND TRIM(upstream_source) != ''
  GROUP BY 1, 2
),

-- Calculate totals for sizing and get display names
primary_totals AS (
  SELECT 
    base_data.primary_knowledge_source,
    COALESCE(infores.catalog.name, base_data.primary_knowledge_source) as display_name,
    SUM(edge_count) as total_from_primary
  FROM base_data
  LEFT JOIN infores.catalog ON infores.catalog.id = base_data.primary_knowledge_source
  GROUP BY base_data.primary_knowledge_source, infores.catalog.name
),

upstream_totals AS (
  SELECT
    clean_upstream_source,
    SUM(edge_count) as total_from_upstream
  FROM base_data
  GROUP BY clean_upstream_source
),

-- Get unique edge count to unified KG
unified_total AS (
  SELECT 
    SUM(count) as total_edges
  FROM bq.merged_kg_edges
  WHERE primary_knowledge_source IN ${inputs.selected_primary_sources.value}
)

-- Network nodes and links with consistent column structure
SELECT 
  'node' as type, 
  CASE 
    WHEN '${inputs.view_mode}' = 'detailed' THEN primary_knowledge_source
    WHEN total_from_primary < ${smallSourceThreshold} THEN 'Other (Small Sources)'
    ELSE primary_knowledge_source 
  END as node_id,
  CASE 
    WHEN '${inputs.view_mode}' = 'detailed' THEN display_name
    WHEN total_from_primary < ${smallSourceThreshold} THEN 'Other (Small Sources)'
    ELSE display_name 
  END as node_name,
  'primary' as category,
  0 as x_position,
  total_from_primary as value,
  NULL as source,
  NULL as target
FROM primary_totals 

UNION ALL

SELECT 
  'node' as type,
  clean_upstream_source as node_id,
  clean_upstream_source as node_name,
  'aggregator' as category, 
  1 as x_position,
  total_from_upstream as value,
  NULL as source,
  NULL as target
FROM upstream_totals

UNION ALL

SELECT 
  'node' as type,
  'Unified KG' as node_id,
  'Unified KG' as node_name,
  'unified' as category,
  2 as x_position, 
  total_edges as value,
  NULL as source,
  NULL as target
FROM unified_total

UNION ALL

SELECT 
  'link' as type,
  NULL as node_id,
  NULL as node_name,
  NULL as category,
  NULL as x_position,
  edge_count as value,
  CASE 
    WHEN '${inputs.view_mode}' = 'detailed' THEN primary_knowledge_source
    WHEN (SELECT total_from_primary FROM primary_totals pt WHERE pt.primary_knowledge_source = bd.primary_knowledge_source) < ${smallSourceThreshold} THEN 'Other (Small Sources)'
    ELSE primary_knowledge_source 
  END as source,
  clean_upstream_source as target
FROM base_data bd

UNION ALL

SELECT 
  'link' as type,
  NULL as node_id,
  NULL as node_name,
  NULL as category,
  NULL as x_position,
  SUM(edge_count) as value,
  clean_upstream_source as source,
  'Unified KG' as target
FROM base_data
GROUP BY clean_upstream_source
```

<KnowledgeSourceFlowGraph 
  networkData={network_data} 
  title="Knowledge Source Flow Network"
  topNPrimarySources={TOP_N_PRIMARY_SOURCES}
  height="900px" 
/>