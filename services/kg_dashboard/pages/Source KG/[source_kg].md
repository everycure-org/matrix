# {params.source_kg}

## Source KG Information

```sql source_kg_info
SELECT 
  upstream_data_source,
  n_nodes,
  n_edges,
  ROUND(n_edges * 1.0 / n_nodes, 2) as edges_per_node
FROM bq.upstream_data_sources
WHERE upstream_data_source = '${params.source_kg}'
```

{#if source_kg_info.length > 0}

<Grid col=3 class="max-w-4xl mx-auto mb-6">
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={source_kg_info} column="n_nodes" fmt="num0" />
    </span><br/>
    Nodes
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={source_kg_info} column="n_edges" fmt="num0" />
    </span><br/>
    Edges
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={source_kg_info} column="edges_per_node" fmt="num2" />
    </span><br/>
    Edges per Node
  </div>
</Grid>

## Edge Categories Distribution

```sql source_kg_sankey
-- First level: Subject Category to Predicate
SELECT 
    concat('[S] ', replace(subject_category,'biolink:','')) as source,
    replace(predicate,'biolink:','') as target,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE upstream_data_source LIKE '%${params.source_kg}%'
GROUP BY all

UNION ALL

-- Second level: Predicate to Object Category
SELECT 
    replace(predicate,'biolink:','') as source,
    concat('[O] ', replace(object_category,'biolink:','')) as target,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE upstream_data_source LIKE '%${params.source_kg}%'
GROUP BY all
ORDER BY count DESC
```

{#if source_kg_sankey.length > 0}
<SankeyDiagram data={source_kg_sankey} 
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  title='Source KG Edge Flow'
  subtitle='Flow from Subject Categories through Predicates to Object Categories'
  chartAreaHeight={800}
/>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No edge data found for source KG "{params.source_kg}".
</div>
{/if}

## Node Categories

```sql source_kg_nodes
SELECT 
  replace(category,'biolink:','') as category,
  sum(count) as node_count
FROM bq.merged_kg_nodes
WHERE upstream_data_source LIKE '%${params.source_kg}%'
GROUP BY category
ORDER BY node_count DESC
LIMIT 20
```

{#if source_kg_nodes.length > 0}
<DataTable data={source_kg_nodes} search=true>
  <Column id="category" title="Node Category" />
  <Column id="node_count" title="Node Count" fmt="num0" />
</DataTable>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No node data found for source KG "{params.source_kg}".
</div>
{/if}

## Edge Type Validation

```sql source_kg_edge_validation
WITH edge_validation AS (
  SELECT 
    edges.subject_category,
    edges.predicate,
    edges.object_category,
    edges.count AS edge_count,
    CASE 
      WHEN valid_types.subject_category IS NOT NULL THEN 'Recognized'
      ELSE 'Unrecognized'
    END AS validation_status
  FROM bq.merged_kg_edges AS edges
  LEFT JOIN valid_edge_types.valid_edge_types AS valid_types
    ON edges.subject_category = valid_types.subject_category
    AND edges.predicate = valid_types.predicate  
    AND edges.object_category = valid_types.object_category
  WHERE edges.upstream_data_source LIKE '%${params.source_kg}%'
),
totals AS (
  SELECT 
    validation_status,
    SUM(edge_count) as total_edges
  FROM edge_validation
  GROUP BY validation_status
),
percentages AS (
  SELECT 
    validation_status,
    total_edges,
    ROUND(100.0 * total_edges / SUM(total_edges) OVER (), 1) AS percentage
  FROM totals
)
SELECT 
  validation_status,
  total_edges,
  percentage
FROM percentages
ORDER BY validation_status DESC
```

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  This section shows the validation of edge types in this source KG against the recognized biolink schema patterns.
</div>

{#if source_kg_edge_validation.length > 0}
<Grid col=2 class="max-w-4xl mx-auto mb-6">
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #1e40af;">
      {#if source_kg_edge_validation.filter(d => d.validation_status === 'Recognized').length > 0}
        <Value data={source_kg_edge_validation.filter(d => d.validation_status === 'Recognized')} column="total_edges" fmt="num0" />
      {:else}
        0
      {/if}
    </span><br/>
    <span class="text-xl">Recognized Edges</span><br/>
    <span class="text-lg" style="color: #1e40af;">
      {#if source_kg_edge_validation.filter(d => d.validation_status === 'Recognized').length > 0}
        (<Value data={source_kg_edge_validation.filter(d => d.validation_status === 'Recognized')} column="percentage" fmt="num1" />%)
      {:else}
        (0.0%)
      {/if}
    </span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #93c5fd;">
      {#if source_kg_edge_validation.filter(d => d.validation_status === 'Unrecognized').length > 0}
        <Value data={source_kg_edge_validation.filter(d => d.validation_status === 'Unrecognized')} column="total_edges" fmt="num0" />
      {:else}
        0
      {/if}
    </span><br/>
    <span class="text-xl">Unrecognized Edges</span><br/>
    <span class="text-lg" style="color: #93c5fd;">
      {#if source_kg_edge_validation.filter(d => d.validation_status === 'Unrecognized').length > 0}
        (<Value data={source_kg_edge_validation.filter(d => d.validation_status === 'Unrecognized')} column="percentage" fmt="num1" />%)
      {:else}
        (0.0%)
      {/if}
    </span>
  </div>
</Grid>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No validation data available for source KG "{params.source_kg}".
</div>
{/if}

## Most Connected Nodes

```sql source_kg_top_nodes
SELECT 
  SPLIT(id, ':')[OFFSET(0)] as prefix,
  replace(category,'biolink:','') as category,
  sum(count) as total_connections
FROM bq.merged_kg_nodes
WHERE upstream_data_source LIKE '%${params.source_kg}%'
GROUP BY id, category
ORDER BY total_connections DESC
LIMIT 10
```

{#if source_kg_top_nodes.length > 0}
<DataTable data={source_kg_top_nodes}>
  <Column id="prefix" title="Node Prefix" />
  <Column id="category" title="Category" />
  <Column id="total_connections" title="Connections" fmt="num0" />
</DataTable>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No connectivity data found for source KG "{params.source_kg}".
</div>
{/if}

{:else}
<div class="text-center text-lg text-red-500 mt-10">
  Source KG "{params.source_kg}" not found in upstream data sources.
</div>
{/if}