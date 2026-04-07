<script>
  import { drugColor, diseaseColor, getCategoryColor } from '../../../_lib/colors';
</script>

```sql component_info
SELECT *
FROM bq.connected_components_detail
WHERE component_id = ${params.component_id}
```

```sql component_nodes
SELECT
  id,
  name,
  category,
  ec_core_category
FROM bq.component_nodes
WHERE component_id = ${params.component_id}
ORDER BY
  CASE WHEN ec_core_category IS NOT NULL THEN 0 ELSE 1 END,
  category, name
```

```sql component_edges
SELECT
  subject_name,
  subject_category,
  predicate,
  object_name,
  object_category,
  primary_knowledge_source
FROM bq.component_edges
WHERE component_id = ${params.component_id}
ORDER BY subject_category, predicate, object_category
```

```sql node_categories
SELECT
  category,
  COUNT(*) as count
FROM bq.component_nodes
WHERE component_id = ${params.component_id}
GROUP BY category
ORDER BY count DESC
```

```sql core_nodes
SELECT
  CASE
    WHEN ec_core_category = 'drug' THEN 'Core Drugs'
    WHEN ec_core_category = 'disease' THEN 'Core Diseases'
  END as label,
  COUNT(*) as count
FROM bq.component_nodes
WHERE component_id = ${params.component_id}
  AND ec_core_category IS NOT NULL
GROUP BY label
ORDER BY count DESC
```

```sql noncore_nodes
SELECT
  parent_category,
  COUNT(*) as count
FROM bq.component_nodes
WHERE component_id = ${params.component_id}
  AND ec_core_category IS NULL
GROUP BY parent_category
ORDER BY count DESC
```

```sql knowledge_sources
SELECT
  primary_knowledge_source,
  COUNT(*) as edge_count
FROM bq.component_edges
WHERE component_id = ${params.component_id}
GROUP BY primary_knowledge_source
ORDER BY edge_count DESC
```

{#if component_info.length > 0}

<a href="/Metrics/connected-components" class="text-sm text-blue-600 hover:text-blue-800 no-underline">&larr; Back to Connected Components</a>

# Component {parseInt(params.component_id)}

<Grid col=4 class="max-w-4xl mx-auto mb-8 mt-6">
  <div class="text-center">
    <span class="font-semibold text-3xl">
      <Value data={component_info} column="component_size" fmt="num0" />
    </span><br/>
    <span class="text-base">Nodes</span>
  </div>
  <div class="text-center">
    <span class="font-semibold text-3xl">
      {component_edges.length.toLocaleString()}
    </span><br/>
    <span class="text-base">Edges</span>
  </div>
  <div class="text-center">
    <span class="font-semibold text-3xl" style="color: {drugColor};">
      <Value data={component_info} column="num_drugs" fmt="num0" />
    </span><br/>
    <span class="text-base">Core Drugs</span>
  </div>
  <div class="text-center">
    <span class="font-semibold text-3xl" style="color: {diseaseColor};">
      <Value data={component_info} column="num_diseases" fmt="num0" />
    </span><br/>
    <span class="text-base">Core Diseases</span>
  </div>
</Grid>

{#if knowledge_sources.length > 0}

## Knowledge Sources

<div class="max-w-3xl mx-auto mb-8">
  {#each knowledge_sources as ks, i}
    <a href="/Knowledge Sources/infores:{ks.primary_knowledge_source}" class="inline-block text-sm bg-gray-100 rounded px-2 py-0.5 mr-1 mb-1 text-blue-600 hover:bg-gray-200 no-underline">{ks.primary_knowledge_source} <span class="text-gray-400">({ks.edge_count})</span></a>
  {/each}
</div>
{/if}

## Composition

{#if core_nodes.length > 0}
<Grid col=2 class="max-w-4xl mx-auto mb-8">
  <div>
    <ECharts config={{
        title: {
            text: 'Core Entities',
            left: 'center',
            top: 'center',
            textStyle: {
                fontWeight: 'normal'
            }
        },
        color: core_nodes.map(row => row.label === 'Core Drugs' ? drugColor : diseaseColor),
        tooltip: {
            position: 'right',
            confine: true,
            formatter: function(params) {
                const count = params.data.value.toLocaleString();
                return `${params.name}: ${count} nodes (${params.percent}%)`;
            }
        },
        series: [{
            type: 'pie',
            data: core_nodes.map(row => ({
              value: row.count,
              name: row.label
            })),
            radius: ['40%', '65%']
        }]
    }}/>
  </div>
  <div class="text-center flex flex-col justify-center">
    {#each core_nodes as row}
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: {row.label === 'Core Drugs' ? drugColor : diseaseColor};">
        {row.count.toLocaleString()}
      </span><br/>
      <span class="text-base">{row.label}</span>
    </div>
    {/each}
    <div class="mb-3">
      <span class="text-xs text-gray-600">
        {core_nodes.reduce((s, r) => s + r.count, 0).toLocaleString()} core entities of {component_nodes.length.toLocaleString()} total nodes
      </span>
    </div>
  </div>
</Grid>
{/if}

{#if noncore_nodes.length > 0}
<Grid col=2 class="max-w-4xl mx-auto mb-8">
  <div>
    <ECharts config={{
        title: {
            text: core_nodes.length > 0 ? 'Non-Core\nNodes' : 'Nodes by\nCategory',
            left: 'center',
            top: 'center',
            textStyle: {
                fontWeight: 'normal',
                fontSize: 13
            }
        },
        color: noncore_nodes.map(row => getCategoryColor(row.parent_category)),
        tooltip: {
            position: 'right',
            confine: true,
            formatter: function(params) {
                const count = params.data.value.toLocaleString();
                return `${params.name}: ${count} nodes (${params.percent}%)`;
            }
        },
        series: [{
            type: 'pie',
            data: noncore_nodes.map(row => ({
              value: row.count,
              name: row.parent_category
            })),
            radius: ['40%', '65%'],
            label: {
                formatter: function(params) {
                    return params.percent >= 3 ? params.name : '';
                }
            }
        }]
    }}/>
  </div>
  <div>
    <DataTable data={noncore_nodes} rows=15>
      <Column id="parent_category" title="Category" />
      <Column id="count" title="Nodes" fmt="num0" contentType="bar" />
    </DataTable>
  </div>
</Grid>
{/if}

## Nodes

<DataTable data={component_nodes} rows=20 search=true>
  <Column id="name" title="Name" />
  <Column id="id" title="ID" />
  <Column id="category" title="Category" />
  <Column id="ec_core_category" title="Core Entity" />
</DataTable>

## Edges

{#if component_edges.length > 0}
<DataTable data={component_edges} rows=20 search=true>
  <Column id="subject_name" title="Subject" />
  <Column id="subject_category" title="Subject Category" />
  <Column id="predicate" title="Predicate" />
  <Column id="object_name" title="Object" />
  <Column id="object_category" title="Object Category" />
  <Column id="primary_knowledge_source" title="Knowledge Source" />
</DataTable>
{:else}
<div class="max-w-md mx-auto text-center mb-8 p-3 bg-gray-50 rounded">
  <span class="text-gray-600">This is a singleton node with no edges.</span>
  <div class="text-xs text-gray-500 mt-1">
    The node exists in the node table but has no connections in the edge table.
  </div>
</div>
{/if}

{:else}

<div class="text-center text-lg text-gray-500 mt-10">
  No data available for component {params.component_id}. Detail data is available for the 50 largest minor components and any components containing core entities.
</div>

{/if}
