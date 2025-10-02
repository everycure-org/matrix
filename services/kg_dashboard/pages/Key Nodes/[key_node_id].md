# {key_node_info.length > 0 ? key_node_info[0].name || params.key_node_id : params.key_node_id}

```sql key_node_info
SELECT * FROM bq.key_nodes_stats WHERE id = '${params.key_node_id}'
```

```sql key_node_edges_breakdown
SELECT * FROM bq.key_nodes_edges_breakdown WHERE key_node_id = '${params.key_node_id}' ORDER BY edge_count DESC
```

```sql key_node_connected_categories
SELECT * FROM bq.key_nodes_connected_categories WHERE key_node_id = '${params.key_node_id}' ORDER BY count DESC
```

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
      <Value data={key_node_info} column="with_descendants_total_edges" fmt="num0" />
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
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl" style="color: #7c3aed;">
      <Value data={key_node_info} column="with_descendants_unique_neighbors" fmt="num0" />
    </span><br/>
    <span class="text-md">Unique Neighbors (with descendants)</span>
  </div>
</Grid>

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
    
    <Column id="subject_category" title="Subject Category" />
    <Column id="predicate" title="Predicate" />
    <Column id="object_category" title="Object Category" />
    <Column id="primary_knowledge_source" title="Primary KS" />
    <Column id="edge_count" title="Edge Count" contentType="bar" fmt="num0" />
    <Column id="unique_subjects" title="Unique Subjects" fmt="num0" />
    <Column id="unique_objects" title="Unique Objects" fmt="num0" />
</DataTable>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No edge data found for this key node.
</div>
{/if}

{:else}
<div class="text-center text-lg text-red-500 mt-10">
  Key node "{params.key_node_id}" not found in the knowledge graph.
</div>
{/if}
