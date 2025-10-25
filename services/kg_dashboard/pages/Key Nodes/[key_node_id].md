# {key_node_info.length > 0 ? key_node_info[0].name || params.key_node_id : params.key_node_id}

```sql key_node_info
SELECT * FROM bq.key_nodes_stats WHERE id = '${params.key_node_id}'
```

```sql key_node_aggregate
SELECT * FROM bq.key_nodes_release_aggregate
WHERE key_node_id = '${params.key_node_id}'
  AND bq_version = '${import.meta.env.VITE_release_version?.replace(/\./g, '_')}'
```

```sql key_node_edges_breakdown
SELECT
  key_node_id,
  REPLACE(subject_category, 'biolink:', '') as subject_category,
  REPLACE(predicate, 'biolink:', '') as predicate,
  REPLACE(object_category, 'biolink:', '') as object_category,
  edge_count
FROM bq.key_nodes_edges_breakdown
WHERE key_node_id = '${params.key_node_id}'
ORDER BY edge_count DESC
```

```sql key_node_connected_categories
SELECT * FROM bq.key_nodes_connected_categories WHERE key_node_id = '${params.key_node_id}' ORDER BY count DESC
```

```sql key_node_category_summary
SELECT * FROM bq.key_nodes_category_summary WHERE key_node_id = '${params.key_node_id}' ORDER BY distinct_nodes DESC
```

```sql key_node_category_edges
SELECT
  parent_category,
  subject,
  subject_name,
  predicate,
  object,
  object_name,
  primary_knowledge_sources
FROM bq.key_nodes_category_edges
WHERE key_node_id = '${params.key_node_id}'
ORDER BY parent_category, subject, predicate, object
```

```sql key_node_release_trends
SELECT
  key_node_id,
  bq_version,
  semantic_version,
  release_order,
  REPLACE(predicate, 'biolink:', '') as predicate,
  REPLACE(subject_category, 'biolink:', '') as subject_category,
  REPLACE(object_category, 'biolink:', '') as object_category,
  REPLACE(primary_knowledge_source, 'infores:', '') as primary_knowledge_source,
  edge_count,
  unique_subjects,
  unique_objects
FROM bq.key_nodes_release_trends
WHERE key_node_id = '${params.key_node_id}'
ORDER BY release_order, edge_count DESC
```

<script>
  import KeyNodeChordDashboard from '../../_components/KeyNodeChordDashboard.svelte';

  const current_release_bq_version = import.meta.env.VITE_release_version?.replace(/\./g, '_') || 'v0_10_4';
  const benchmark_bq_version = import.meta.env.VITE_benchmark_version?.replace(/\./g, '_') || 'v0_10_2';
  const benchmark_semantic_version = import.meta.env.VITE_benchmark_version || 'v0.10.2';

  // Filter release trends data in JavaScript
  $: current_release_edges = key_node_release_trends.filter(row => row.bq_version === current_release_bq_version);
  $: benchmark_release_edges = key_node_release_trends.filter(row => row.bq_version === benchmark_bq_version);

  // Compute edges added (in current but not in benchmark)
  $: edges_added = current_release_edges.filter(curr => {
    return !benchmark_release_edges.some(bench =>
      bench.predicate === curr.predicate &&
      bench.subject_category === curr.subject_category &&
      bench.object_category === curr.object_category &&
      bench.primary_knowledge_source === curr.primary_knowledge_source
    );
  }).sort((a, b) => b.edge_count - a.edge_count);

  // Compute edges removed (in benchmark but not in current)
  $: edges_removed = benchmark_release_edges.filter(bench => {
    return !current_release_edges.some(curr =>
      curr.predicate === bench.predicate &&
      curr.subject_category === bench.subject_category &&
      curr.object_category === bench.object_category &&
      curr.primary_knowledge_source === bench.primary_knowledge_source
    );
  }).sort((a, b) => b.edge_count - a.edge_count);

  // Compute edges with significant changes
  $: edges_changed = current_release_edges
    .map(curr => {
      const bench = benchmark_release_edges.find(b =>
        b.predicate === curr.predicate &&
        b.subject_category === curr.subject_category &&
        b.object_category === curr.object_category &&
        b.primary_knowledge_source === curr.primary_knowledge_source
      );
      if (bench && Math.abs(curr.edge_count - bench.edge_count) > 10) {
        return {
          ...curr,
          benchmark_count: bench.edge_count,
          current_count: curr.edge_count,
          count_change: curr.edge_count - bench.edge_count,
          pct_change: Math.round(1000 * (curr.edge_count - bench.edge_count) / bench.edge_count) / 10
        };
      }
      return null;
    })
    .filter(x => x !== null)
    .sort((a, b) => Math.abs(b.count_change) - Math.abs(a.count_change));
</script>

{#if key_node_info.length > 0}

<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  <strong>ID:</strong> {key_node_info[0].id}<br/>
  <strong>Category:</strong> {key_node_info[0].category?.replace('biolink:', '') || 'N/A'}
</div>

## Overview Statistics

<Details title="Understanding These Metrics">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
Statistics are shown for both the key node directly and including all its descendants (subtypes/subclasses).
Descendants are found by recursively following biolink:subclass_of edges up to 20 levels deep.
This provides a comprehensive view of the entire hierarchy under this node.
</div>
</Details>

<Grid col=2 class="max-w-4xl mx-auto mb-6">
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #1e40af;">
      <Value data={key_node_info} column="descendant_count" fmt="num0" />
    </span><br/>
    <span class="text-xl">Descendants</span><br/>
    <span class="text-sm text-gray-600">(including self)</span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #1e40af;">
      <Value data={key_node_info} column="max_descendant_depth" fmt="num0" />
    </span><br/>
    <span class="text-xl">Max Hierarchy Depth</span>
  </div>
</Grid>

<Grid col=2 class="max-w-4xl mx-auto mb-6">
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl" style="color: #059669;">
      <Value data={key_node_info} column="direct_total_edges" fmt="num0" />
    </span><br/>
    <span class="text-md">Direct Edges</span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl" style="color: #059669;">
      <Value data={key_node_aggregate} column="with_descendants_total_edges" fmt="num0" />
    </span><br/>
    <span class="text-md">Edges (with descendants)</span>
  </div>
</Grid>

<Grid col=2 class="max-w-4xl mx-auto mb-6">
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl" style="color: #7c3aed;">
      <Value data={key_node_info} column="direct_unique_neighbors" fmt="num0" />
    </span><br/>
    <span class="text-md">Direct Unique Neighbors</span>
  </div>
</Grid>

## Interactive Category Explorer

<Details title="Understanding This Visualization">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This interactive chord diagram shows the key node in the center with connected categories arranged in a circle around it.
The size and color of each link represents the number of edges to that category. Click on any category node to see
detailed edge information below. Click again or click the key node to deselect.
</div>
</Details>

{#if key_node_category_summary.length > 0}
<KeyNodeChordDashboard
  categoryData={key_node_category_summary}
  edgeData={key_node_category_edges}
  keyNodeName={key_node_info.length > 0 ? key_node_info[0].name || params.key_node_id : params.key_node_id}
/>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No category data available for visualization.
</div>
{/if}

## Connection Flow

<Details title="Understanding This Diagram">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This Sankey diagram shows how this key node and its descendants connect to other categories in the knowledge graph.
Incoming connections (labeled [IN]) show categories that connect TO this node, while outgoing connections (labeled [OUT])
show categories this node connects TO. Only connections with more than 100 edges are shown.
</div>
</Details>

{#if key_node_connected_categories.length > 0}
<SankeyDiagram
  data={key_node_connected_categories}
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  title='Key Node Connection Flow (with descendants)'
  subtitle='Flow from Incoming Categories through Key Node to Outgoing Categories (>100 connections)'
  chartAreaHeight={500}
/>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No significant connections found (threshold: >100 edges).
</div>
{/if}

## Edge Type Breakdown

<Details title="Understanding This Table">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This table shows all edge types involving this key node and its descendants, broken down by predicate (relationship type),
subject category, object category, and primary knowledge source. This helps understand what types of relationships
are most common for this entity.
</div>
</Details>

{#if key_node_edges_breakdown.length > 0}
<DataTable
    data={key_node_edges_breakdown}
    search=true
    pagination=true
    pageSize={25}
    title="Edge Types (with descendants)">

    <Column id="edge_count" title="Edge Count" contentType="bar" fmt="num0" />
    <Column id="subject_category" title="Subject Category" />
    <Column id="predicate" title="Predicate" />
    <Column id="object_category" title="Object Category" />

</DataTable>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No edge data found for this key node.
</div>
{/if}

## Release Trends

```sql key_node_aggregate_trends
SELECT * FROM bq.key_nodes_release_aggregate WHERE key_node_id = '${params.key_node_id}' ORDER BY release_order
```

<Details title="Understanding Release Trends">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This section shows how the neighborhood of this key node (including descendants) has evolved across releases.
Direct edges are connections involving only the key node itself, while "with descendants" includes all subtypes/subclasses.
The comparison tables below show edge types that have been added, removed, or changed significantly between the benchmark and current release.
</div>
</Details>

{#if key_node_aggregate_trends.length > 0}

### Edge Counts Over Time

<LineChart
    data={key_node_aggregate_trends}
    x="semantic_version"
    y={['direct_total_edges', 'with_descendants_total_edges']}
    ySeriesLabels={['Direct Edges', 'With Descendants']}
    title="Total Edges Across Releases"
    yGridlines=false
    xBaseline=false
    xAxisLabels=false
    markers=false
    step=true
    sort=false
>
    <ReferenceLine x={benchmark_semantic_version} label="Benchmark" hideValue=true/>
</LineChart>

{/if}

{#if key_node_release_trends.length > 0}

### Edge Types Added (Current vs Benchmark)

{#if edges_added.length > 0}
<DataTable
    data={edges_added}
    search=true
    pagination=true
    pageSize={10}
    title="Edge types present in current release but not in benchmark">
    <Column id="edge_count" title="Edge Count" contentType="bar" fmt="num0" />
    <Column id="subject_category" title="Subject Category" />
    <Column id="predicate" title="Predicate" />
    <Column id="object_category" title="Object Category" />
</DataTable>
{:else}
<div class="text-center text-lg text-gray-500 mt-4 mb-4">
  No new edge types added in current release.
</div>
{/if}

### Edge Types Removed (Current vs Benchmark)

{#if edges_removed.length > 0}
<DataTable
    data={edges_removed}
    search=true
    pagination=true
    pageSize={10}
    title="Edge types present in benchmark but not in current release">
    <Column id="edge_count" title="Edge Count (Benchmark)" contentType="bar" fmt="num0" />
    <Column id="subject_category" title="Subject Category" />
    <Column id="predicate" title="Predicate" />
    <Column id="object_category" title="Object Category" />
</DataTable>
{:else}
<div class="text-center text-lg text-gray-500 mt-4 mb-4">
  No edge types removed in current release.
</div>
{/if}

### Edge Types with Significant Changes

{#if edges_changed.length > 0}
<DataTable
    data={edges_changed}
    search=true
    pagination=true
    pageSize={10}
    title="Edge types with significant count changes (>10 edges difference)">
    <Column id="count_change" title="Change" contentType="delta" fmt="num0" />
    <Column id="pct_change" title="% Change" fmt="pct1" />
    <Column id="benchmark_count" title="Benchmark Count" fmt="num0" />
    <Column id="current_count" title="Current Count" fmt="num0" />
    <Column id="subject_category" title="Subject Category" />
    <Column id="predicate" title="Predicate" />
    <Column id="object_category" title="Object Category" />
</DataTable>
{:else}
<div class="text-center text-lg text-gray-500 mt-4 mb-4">
  No significant changes in edge counts.
</div>
{/if}

{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No release trend data available for this key node.
</div>
{/if}

{:else}
<div class="text-center text-lg text-red-500 mt-10">
  Key node "{params.key_node_id}" not found in the knowledge graph.
</div>
{/if}
