---
title: Knowledge Sources
---

<script>
  import { getSourceColor } from '../_lib/colors';
  
  // Create depth overrides for proper Sankey layout (legacy - keeping for compatibility)
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

  const smallSourceThreshold = 50000;
  
  // Network graph configuration
  let networkOption = {};
  
  $: {
    // Process the network data
    let nodes = [];
    let links = [];
    
    // Debug: log the raw data
    console.log('Network data:', network_data);
    console.log('Network data type:', typeof network_data);
    console.log('Network data length:', network_data?.length);
    console.log('Network data is array:', Array.isArray(network_data));
    
    if (network_data && Array.isArray(network_data) && network_data.length > 0) {
      // Separate nodes and links
      const nodeData = network_data.filter(d => d.type === 'node');
      const linkData = network_data.filter(d => d.type === 'link');
      
      console.log('Node data count:', nodeData.length);
      console.log('Link data count:', linkData.length);
      console.log('Sample node:', nodeData[0]);
      console.log('Sample link:', linkData[0]);
      
      // Create a Map to ensure unique nodes
      const nodeMap = new Map();
      
      // Sort nodes by category and value for better positioning
      const sortedNodes = nodeData.sort((a, b) => {
        if (a.category !== b.category) {
          const categoryOrder = { 'primary': 0, 'aggregator': 1, 'unified': 2 };
          return categoryOrder[a.category] - categoryOrder[b.category];
        }
        return (b.value || 0) - (a.value || 0); // Larger values first
      });
      
      // Adjust primary sources for oval layout
      const primaryNodes = sortedNodes.filter(n => n.category === 'primary').slice(0, 20);
      const aggregatorNodes = sortedNodes.filter(n => n.category === 'aggregator');
      const unifiedNodes = sortedNodes.filter(n => n.category === 'unified');
      const limitedNodes = [...primaryNodes, ...aggregatorNodes, ...unifiedNodes];
      
      // Calculate oval layout for primary sources (left half)
      const radiusX = 150; // Horizontal radius (narrower)
      const radiusY = 250; // Vertical radius (taller for more space)
      const centerX = 400; // Center X position of the oval
      const centerY = 300; // Center Y position of the oval
      const startAngle = Math.PI * 0.7; // Start at ~126 degrees (upper left)
      const endAngle = Math.PI * 1.3; // End at ~234 degrees (lower left) - extends more to the right
      const angleRange = endAngle - startAngle; // Total angle span
      
      // Position aggregators and unified KG relative to the circle
      const aggregatorStartY = centerY - 50; // Position near center of circle
      const aggregatorSpacing = 100;
      const unifiedY = aggregatorNodes.length > 1 ? 
        aggregatorStartY + (aggregatorNodes.length - 1) * aggregatorSpacing / 2 :
        aggregatorStartY;
      
      let primaryCount = 0, aggregatorCount = 0, unifiedCount = 0;
      
      limitedNodes.forEach(d => {
        if (d.node_id && !nodeMap.has(d.node_id)) {
          // Assign non-overlapping positions based on category
          let xPosition = 0, yPosition = 0;
          
          if (d.category === 'primary') {
            // Calculate position on left half oval
            const angle = startAngle + (primaryCount / (primaryNodes.length - 1)) * angleRange;
            xPosition = centerX + Math.cos(angle) * radiusX; // Use horizontal radius
            yPosition = centerY + Math.sin(angle) * radiusY; // Use vertical radius
            primaryCount++;
          } else if (d.category === 'aggregator') {
            xPosition = 350; // Move aggregators much closer to the oval
            yPosition = aggregatorStartY + aggregatorCount * aggregatorSpacing;
            aggregatorCount++;
          } else if (d.category === 'unified') {
            xPosition = 450; // Move unified KG much closer to aggregators
            yPosition = unifiedY;
            unifiedCount++;
          }
          
          // Get color based on node ID and category
          let nodeColor;
          if (d.category === 'primary') {
            nodeColor = '#88C0D0'; // Light blue for primary sources
          } else if (d.category === 'aggregator' || d.category === 'unified') {
            const sourceKey = d.node_id.replace('infores:', '');
            nodeColor = getSourceColor(sourceKey);
          }

          nodeMap.set(d.node_id, {
            id: d.node_id,
            name: d.node_id,
            nodeCategory: d.category, // Store as custom property, not ECharts category
            value: d.value || 0,
            x: xPosition,
            y: yPosition,
            symbol: 'circle', // All circles
            symbolSize: d.category === 'primary' ? 
              Math.max(15, Math.min(45, (d.value || 0) / 1000000 * 0.4 + 15)) : // Primary: same scaling formula as aggregators/unified
              d.category === 'unified' ? 
                Math.max(60, Math.min(100, (d.value || 0) / 1000000 * 0.8 + 60)) : // Unified: largest circles
                Math.max(45, Math.min(80, (d.value || 0) / 1000000 * 0.6 + 45)), // Aggregators: medium circles
            itemStyle: nodeColor ? { color: nodeColor } : undefined,
            label: {
              show: true,
              position: d.category === 'primary' ? 'left' : 'inside',
              fontSize: d.category === 'primary' ? 12 : 11,
              color: d.category === 'primary' ? '#333' : '#fff',
              fontWeight: d.category === 'primary' ? 'normal' : 'bold',
              distance: d.category === 'primary' ? 8 : 0
            },
            fixed: true
          });
        }
      });
      
      nodes = Array.from(nodeMap.values());
      
      // Create links, filtering out invalid ones
      links = linkData
        .filter(d => d.source && d.target && d.value)
        .map(d => {
          // Determine edge color based on aggregator node
          let edgeColor = 'rgba(157, 121, 214, 0.6)'; // default purple
          
          // Find which node is the aggregator by checking if it's robokop or rtxkg2
          const sourceNode = nodes.find(n => n.id === d.source);
          const targetNode = nodes.find(n => n.id === d.target);
          
          let aggregatorNode = null;
          
          // Check if source is an aggregator (robokop or rtxkg2)
          if (sourceNode && (sourceNode.id.includes('robokop') || sourceNode.id.includes('rtxkg2')) && 
              !sourceNode.id.includes(',')) { // Single aggregator, not unified
            aggregatorNode = sourceNode;
          } 
          // Check if target is an aggregator
          else if (targetNode && (targetNode.id.includes('robokop') || targetNode.id.includes('rtxkg2')) && 
                   !targetNode.id.includes(',')) { // Single aggregator, not unified
            aggregatorNode = targetNode;
          }
          
          if (aggregatorNode && aggregatorNode.itemStyle && aggregatorNode.itemStyle.color) {
            edgeColor = aggregatorNode.itemStyle.color + '80'; // Add transparency
          }
          
          return {
            source: d.source,
            target: d.target,
            value: d.value,
            lineStyle: {
              width: Math.max(1, Math.min(10, Math.sqrt(d.value / 10000) * 5 + 1)), // Thickness based on count
              color: edgeColor
            },
            label: {
              show: false,
              position: 'middle',
              formatter: function(params) {
                return params.value.toLocaleString();
              },
              fontSize: 10,
              color: '#333',
              backgroundColor: 'rgba(255, 255, 255, 0.8)',
              borderRadius: 3,
              padding: [2, 4]
            },
            emphasis: {
              label: {
                show: true
              }
            }
          };
        });
      
      console.log('Final nodes count:', nodes.length);
      console.log('Final links count:', links.length);
      console.log('Sample final node:', nodes[0]);
      console.log('Sample final link:', links[0]);
      console.log('Node positions:', nodes.slice(0, 5).map(n => ({id: n.id, category: n.category, x: n.x, y: n.y})));
      console.log('Node values with sizes:', nodes.slice(0, 5).map(n => ({
        id: n.id.substring(8), // Remove 'infores:' prefix for readability
        category: n.category, 
        value: n.value, 
        width: n.symbolSize[0], 
        height: n.symbolSize[1]
      })));
      
      // Specifically log aggregators and unified KG
      const aggregators = nodes.filter(n => n.category === 'aggregator' || n.category === 'unified');
      console.log('Aggregators and Unified KG:', aggregators.map(n => ({
        id: n.id,
        category: n.category,
        value: n.value,
        width: n.symbolSize[0],
        height: n.symbolSize[1]
      })));
      
      // Debug primary source positioning
      console.log('All nodes by category:', nodes.map(n => n.category));
      const primaries = nodes.filter(n => n.category === 'primary');
      console.log('Found', primaries.length, 'primary nodes');
      console.log('First 5 primary positions:', primaries.slice(0, 5).map(n => ({
        id: n.id,
        category: n.category,
        x: n.x,
        y: n.y,
        symbolSize: n.symbolSize
      })));
    }
    
    networkOption = {
      legend: {
        show: false
      },
      title: {
        text: 'Knowledge Source Flow Network',
        left: 'center'
      },
      grid: {
        left: '5%',
        right: '5%',
        top: '10%',
        bottom: '15%',
        containLabel: true
      },
      tooltip: {
        trigger: 'item',
        formatter: function(params) {
          if (params.dataType === 'node') {
            return `<strong>${params.name}</strong><br/>Edges: ${params.value.toLocaleString()}`;
          } else {
            return `<strong>${params.data.source} → ${params.data.target}</strong><br/>Edges: ${params.value.toLocaleString()}`;
          }
        }
      },
      legend: {
        data: ['primary', 'aggregator', 'unified'],
        bottom: 10
      },
      series: [{
        type: 'graph',
        layout: 'none', // Try disabling force layout to use our fixed positions
        // force: {
        //   repulsion: 5000,
        //   gravity: 0.02,
        //   edgeLength: [200, 400],
        //   layoutAnimation: false
        // },
        data: nodes,
        links: links,
        roam: false, // Disable zoom/pan
        draggable: false, // Disable dragging
        itemStyle: {
          borderWidth: 0
        },
        lineStyle: {
          color: 'rgba(157, 121, 214, 0.6)',
          curveness: 0.1 // Less curve for cleaner look
        },
        label: {
          show: true,
          position: 'inside',
          fontSize: 10,
          color: '#000'
        },
        tooltip: {
          show: true,
          position: function(point, params, dom, rect, size) {
            if (params.dataType === 'node' && params.data.category === 'primary') {
              // Position tooltip to the left for primary sources
              return [point[0] - size.contentSize[0] - 20, point[1] - size.contentSize[1] / 2];
            }
            // Default positioning for other nodes
            return [point[0] + 20, point[1] - size.contentSize[1] / 2];
          },
          formatter: function(params) {
            if (params.dataType === 'node' && params.data.category === 'primary') {
              // Find all connections from this primary source to aggregators
              const sourceId = params.data.id;
              const connections = links.filter(link => link.source === sourceId);
              
              let tooltipContent = `<strong>${sourceId.replace('infores:', '')}</strong><br/>`;
              tooltipContent += `Total connections: ${params.data.value.toLocaleString()}`;
              
              if (connections.length > 1) {
                tooltipContent += `<br/><br/>`;
                connections.forEach(conn => {
                  const targetName = conn.target.replace('infores:', '');
                  tooltipContent += `${targetName}: ${conn.value.toLocaleString()}<br/>`;
                });
              }
              
              return tooltipContent;
            } else if (params.dataType === 'node') {
              // Default tooltip for other nodes
              return `<strong>${params.data.id.replace('infores:', '')}</strong><br/>Total connections: ${params.data.value.toLocaleString()}`;
            } else if (params.dataType === 'edge') {
              // Tooltip for edges
              return `${params.data.source.replace('infores:', '')} → ${params.data.target.replace('infores:', '')}<br/>Connections: ${params.data.value.toLocaleString()}`;
            }
          }
        },
        emphasis: {
          focus: 'adjacency',
          itemStyle: {
            borderWidth: 3
          },
          lineStyle: {
            width: 4
          }
        }
      }]
    };
    
    console.log('Network option created:', networkOption);
    console.log('Chart data length:', networkOption.series[0].data?.length);
    console.log('Chart links length:', networkOption.series[0].links?.length);
  }
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

**Understanding the visualization:**
- **Primary Sources** (left): Original databases (e.g., DrugCentral, ChEMBL) containing assertions
- **Aggregators** (center): Knowledge graphs (RTX-KG2, ROBOKOP) that ingest and process primary sources 
- **Unified KG** (right): Our merged knowledge graph that combines assertions from all aggregators
- **Edge thickness**: Proportional to the number of connections between sources

**How overlaps are handled:**
- Edge counts show total contributions from each primary source to each aggregator
- The final count into Unified KG represents unique assertions (no double-counting)
- Hover over connections to see detailed edge counts and overlap information

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

-- Calculate totals for sizing
primary_totals AS (
  SELECT 
    primary_knowledge_source,
    SUM(edge_count) as total_from_primary
  FROM base_data
  GROUP BY primary_knowledge_source
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
  NULL as category,
  NULL as x_position,
  SUM(edge_count) as value,
  clean_upstream_source as source,
  'Unified KG' as target
FROM base_data
GROUP BY clean_upstream_source
```

<ECharts config={networkOption} data={network_data} height="800px" width="100%" />