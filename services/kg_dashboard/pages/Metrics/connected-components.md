---
title: Connected Components
---

<script>
  import { CATEGORY_COLORS } from '../../_lib/colors';

  const drugColor = '#51AFA6';
  const diseaseColor = '#C56492';
  const otherColor = '#999999';

  function getCategoryColor(name) {
    return CATEGORY_COLORS[name] || otherColor;
  }
</script>

```sql lcc_headline
SELECT
  SUM(total_nodes) as total_nodes,
  MAX(CASE WHEN size_category = 'LCC' THEN total_nodes END) as lcc_size,
  ROUND(100.0 * MAX(CASE WHEN size_category = 'LCC' THEN total_nodes END) / SUM(total_nodes), 1) as lcc_pct,
  SUM(num_components) - 1 as disconnected_fragments,
  SUM(num_components) as total_components
FROM bq.connected_components
```

```sql core_connectivity
SELECT
  category,
  total_core_entities,
  core_entities_in_lcc,
  ROUND(100.0 * lcc_fraction, 2) as lcc_pct,
  ROUND(weighted_connectivity_score, 4) as weighted_score
FROM bq.core_connectivity_summary
ORDER BY CASE category WHEN 'all_core' THEN 1 WHEN 'drugs' THEN 2 WHEN 'diseases' THEN 3 END
```

```sql lcc_core
SELECT parent_category, node_count
FROM bq.lcc_composition
WHERE is_core = true
ORDER BY node_count DESC
```

```sql lcc_core_total
SELECT SUM(node_count) as total FROM bq.lcc_composition WHERE is_core = true
```

```sql lcc_noncore
WITH ranked AS (
  SELECT parent_category, node_count,
    ROW_NUMBER() OVER (ORDER BY node_count DESC) as rn
  FROM bq.lcc_composition
  WHERE is_core = false
)
SELECT
  CASE WHEN rn <= 7 THEN parent_category ELSE 'Other' END as parent_category,
  SUM(node_count) as node_count
FROM ranked
GROUP BY CASE WHEN rn <= 7 THEN parent_category ELSE 'Other' END
ORDER BY node_count DESC
```

```sql lcc_noncore_total
SELECT SUM(node_count) as total FROM bq.lcc_composition WHERE is_core = false
```

```sql lcc_total
SELECT SUM(node_count) as total FROM bq.lcc_composition
```

```sql size_distribution
SELECT
  size_category,
  num_components,
  total_nodes,
  core_drugs,
  core_diseases
FROM bq.connected_components
ORDER BY sort_order
```

```sql minor_components
SELECT * FROM bq.connected_components_minor
```

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  Connected components measure whether all nodes in the knowledge graph can reach each other through
  paths. For drug repurposing, this matters directly: a drug can only be linked to a disease if they
  exist in the same connected component. The Largest Connected Component (LCC) is the single giant
  cluster that, in a well-integrated graph, contains the vast majority of nodes.
</div>

<Details title="About Connected Components">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  A connected component is a maximal subgraph in which every node can be reached from every other
  node by following edges. Component rank is assigned by size: rank 0 is the LCC, rank 1 is the
  second-largest, and so on.
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  High fragmentation — many small disconnected components — means the graph contains isolated pockets
  of knowledge that cannot contribute to path-based reasoning or link prediction. Ideally, all core
  entities (drugs and diseases from the EC curated lists) should reside in the LCC.
</div>
</Details>

{#if lcc_headline.length > 0}
<Grid col=3 class="max-w-4xl mx-auto mb-8 mt-6">
  <div class="text-center">
    <span class="font-semibold text-4xl">
      <Value data={lcc_headline} column="lcc_pct" fmt="num1" />%
    </span><br/>
    <span class="text-base">of all nodes in the LCC</span>
  </div>
  <div class="text-center">
    <span class="font-semibold text-4xl">
      <Value data={lcc_headline} column="lcc_size" fmt="num0" />
    </span><br/>
    <span class="text-base">nodes in the LCC</span>
  </div>
  <div class="text-center">
    <span class="font-semibold text-4xl">
      <Value data={lcc_headline} column="disconnected_fragments" fmt="num0" />
    </span><br/>
    <span class="text-base">disconnected fragments</span>
  </div>
</Grid>
{/if}

## Core Entity Coverage

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  For drug repurposing to work, both drugs and diseases must be reachable within the graph. These
  numbers show what fraction of EC-curated core entities reside in the LCC.
</div>

{#if core_connectivity.length > 0}
<Grid col=3 class="max-w-4xl mx-auto mb-6">
  <div class="text-center">
    <span class="font-semibold text-3xl">
      <Value data={core_connectivity.filter(d => d.category === 'all_core')} column="lcc_pct" fmt="num2" />%
    </span><br/>
    <span class="text-base font-medium">All Core Entities</span><br/>
    <span class="text-xs text-gray-600">
      <Value data={core_connectivity.filter(d => d.category === 'all_core')} column="core_entities_in_lcc" fmt="num0" /> of <Value data={core_connectivity.filter(d => d.category === 'all_core')} column="total_core_entities" fmt="num0" /> in LCC
    </span>
  </div>
  <div class="text-center">
    <span class="font-semibold text-3xl">
      <Value data={core_connectivity.filter(d => d.category === 'drugs')} column="lcc_pct" fmt="num2" />%
    </span><br/>
    <span class="text-base font-medium">Core Drugs</span><br/>
    <span class="text-xs text-gray-600">
      <Value data={core_connectivity.filter(d => d.category === 'drugs')} column="core_entities_in_lcc" fmt="num0" /> of <Value data={core_connectivity.filter(d => d.category === 'drugs')} column="total_core_entities" fmt="num0" /> in LCC
    </span>
  </div>
  <div class="text-center">
    <span class="font-semibold text-3xl">
      <Value data={core_connectivity.filter(d => d.category === 'diseases')} column="lcc_pct" fmt="num2" />%
    </span><br/>
    <span class="text-base font-medium">Core Diseases</span><br/>
    <span class="text-xs text-gray-600">
      <Value data={core_connectivity.filter(d => d.category === 'diseases')} column="core_entities_in_lcc" fmt="num0" /> of <Value data={core_connectivity.filter(d => d.category === 'diseases')} column="total_core_entities" fmt="num0" /> in LCC
    </span>
  </div>
</Grid>

<div class="max-w-md mx-auto text-center mb-8 p-3 bg-gray-50 rounded">
  <span class="text-sm text-gray-600">Weighted Connectivity Score</span><br/>
  <span class="font-semibold text-xl">
    <Value data={core_connectivity.filter(d => d.category === 'all_core')} column="weighted_score" fmt="num4" />
  </span>
  <div class="text-xs text-gray-500 mt-1">
    Accounts for both core entity placement and component sizes relative to the LCC.
    A score of 1.0 means all core entities are in the LCC.
  </div>
</div>
{/if}

## LCC Composition

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  The LCC contains two kinds of nodes: the core entities we care about for drug repurposing (EC-curated
  drugs and diseases), and all the other biological entities that connect them. Understanding what
  makes up the non-core majority helps assess whether the graph's connective tissue is biologically
  meaningful.
</div>

### Core Entities

{#if lcc_core.length > 0}
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
        color: [drugColor, diseaseColor],
        tooltip: {
            formatter: function(params) {
                const count = params.data.value.toLocaleString();
                return `${params.name}: ${count} nodes (${params.percent}%)`;
            }
        },
        series: [{
            type: 'pie',
            data: lcc_core.map(row => ({
              value: row.node_count,
              name: row.parent_category
            })),
            radius: ['40%', '65%']
        }]
    }}/>
  </div>
  <div class="text-center flex flex-col justify-center">
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: {drugColor};">
        <Value data={lcc_core.filter(d => d.parent_category === 'Core Drugs')} column="node_count" fmt="num0" />
      </span><br/>
      <span class="text-base">Core Drugs</span>
    </div>
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: {diseaseColor};">
        <Value data={lcc_core.filter(d => d.parent_category === 'Core Diseases')} column="node_count" fmt="num0" />
      </span><br/>
      <span class="text-base">Core Diseases</span>
    </div>
    <div class="mb-3">
      <span class="text-xs text-gray-600">
        <Value data={lcc_core_total} column="total" fmt="num0" /> core entities of <Value data={lcc_total} column="total" fmt="num0" /> total LCC nodes
      </span>
    </div>
  </div>
</Grid>
{/if}

### Non-Core Nodes by Category

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  The remaining <Value data={lcc_noncore_total} column="total" fmt="num0" /> non-core nodes provide the
  connective structure between drugs and diseases. The breakdown by biolink parent category shows what
  kinds of biological entities form this bridge.
</div>

{#if lcc_noncore.length > 0}
<Grid col=2 class="max-w-4xl mx-auto mb-8">
  <div>
    <ECharts config={{
        title: {
            text: 'Non-Core\nNodes',
            left: 'center',
            top: 'center',
            textStyle: {
                fontWeight: 'normal',
                fontSize: 13
            }
        },
        color: lcc_noncore.map(row => getCategoryColor(row.parent_category)),
        tooltip: {
            formatter: function(params) {
                const count = params.data.value.toLocaleString();
                return `${params.name}: ${count} nodes (${params.percent}%)`;
            }
        },
        series: [{
            type: 'pie',
            data: lcc_noncore.map(row => ({
              value: row.node_count,
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
    <DataTable data={lcc_noncore} rows=15>
      <Column id="parent_category" title="Category" />
      <Column id="node_count" title="Nodes" fmt="num0" contentType="bar" />
    </DataTable>
  </div>
</Grid>
{/if}

## Component Size Distribution

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  Connected components follow a power-law distribution: the LCC is enormous, and everything else is
  tiny. The table below buckets components by size to show this pattern clearly.
</div>

<DataTable data={size_distribution} rows=10>
  <Column id="size_category" title="Size Category" />
  <Column id="num_components" title="# Components" fmt="num0" contentType="bar" />
  <Column id="total_nodes" title="Total Nodes" fmt="num0" contentType="bar" />
  <Column id="core_drugs" title="Core Drugs" fmt="num0" />
  <Column id="core_diseases" title="Core Diseases" fmt="num0" />
</DataTable>

## Minor Components Detail

<Details title="Show top 20 minor components">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  These are the largest components outside the LCC. The component hash can be used to track
  whether a fragment persists, splits, or merges between releases.
</div>

<DataTable data={minor_components} rows=20 search=true>
  <Column id="component_id" title="Rank" />
  <Column id="component_size" title="Size" fmt="num0" contentType="bar" />
  <Column id="num_drugs" title="Core Drugs" fmt="num0" />
  <Column id="num_diseases" title="Core Diseases" fmt="num0" />
  <Column id="num_other" title="Other" fmt="num0" />
  <Column id="component_hash" title="Hash" />
</DataTable>
</Details>
