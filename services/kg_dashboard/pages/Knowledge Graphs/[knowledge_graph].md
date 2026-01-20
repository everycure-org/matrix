<script>
  import KnowledgeSourceFlowGraph from '../../_components/KnowledgeSourceFlowGraph.svelte';

  const TOP_N_PRIMARY_SOURCES = 25;
  const smallSourceThreshold = 50000;

  const TWO_LEVEL_CONFIG = [
    {
      name: 'primary',
      layout: 'arc',
      label: { position: 'left', fontSize: 12, fontWeight: 'normal', distance: 8 },
      arc: {
        spread: Math.PI * 0.6,
        center: Math.PI,
        radiusScaling: { method: 'logarithmic', reference: 10, minScale: 0.5, maxScale: 1.0 },
        spreadScaling: { method: 'logarithmic', reference: 25, minScale: 0.4, maxScale: 1.0 }
      }
    },
    {
      name: 'aggregator',
      layout: 'vertical',
      label: { position: 'inside', fontSize: 11, fontWeight: 'bold', distance: 0 }
    }
  ];
</script>

```sql kg_info
select * from bq.kg_list where knowledge_graph = '${params.knowledge_graph}'
```

# {kg_info.length > 0 ? kg_info[0].display_name : params.knowledge_graph}

{#if kg_info.length === 0}
<div class="text-center text-lg text-red-500 mt-10">
  Knowledge graph "{params.knowledge_graph}" not found.
</div>
{:else}

```sql pipeline_metrics
select * from bq.kg_pipeline_metrics
where knowledge_graph = '${params.knowledge_graph}'
order by entity_type, sort_order
```

```sql summary_metrics
select * from bq.kg_summary
where knowledge_graph = '${params.knowledge_graph}'
```

```sql top_categories
select
  category_pair,
  subject_category,
  object_category,
  edge_count
from bq.kg_categories
where knowledge_graph = '${params.knowledge_graph}'
order by edge_count desc
limit 10
```

```sql top_predicates
select
  predicate_display,
  predicate,
  edge_count
from bq.kg_predicates
where knowledge_graph = '${params.knowledge_graph}'
order by edge_count desc
limit 10
```

## Overview

<Grid cols=4>
  <BigValue data={summary_metrics} value="node_count" title="Nodes" fmt="num0" />
  <BigValue data={summary_metrics} value="edge_count" title="Edges" fmt="num0" />
  <BigValue data={summary_metrics} value="normalization_success_rate" title="Normalization Rate" fmt="pct1" />
  <BigValue data={summary_metrics} value="provenance_rate" title="Provenance Rate" fmt="pct1" />
</Grid>

## Knowledge Source Flow

The diagram below shows how knowledge flows from primary sources into {kg_info[0].display_name}.

```sql flow_nodes
-- Get data for network visualization - NODES
WITH base_data AS (
  SELECT
    primary_knowledge_source,
    TRIM(upstream_source) as clean_upstream_source,
    SUM(count) as edge_count
  FROM bq.merged_kg_edges
  CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
  WHERE TRIM(upstream_source) = '${params.knowledge_graph}'
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

-- Get unfiltered totals for context
all_upstream_totals AS (
  SELECT
    TRIM(upstream_source) as clean_upstream_source,
    SUM(count) as total_all_sources
  FROM bq.merged_kg_edges
  CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
  WHERE TRIM(upstream_source) = '${params.knowledge_graph}'
  GROUP BY 1
)

-- Primary source nodes
SELECT
  CASE
    WHEN total_from_primary < ${smallSourceThreshold} THEN 'Other (Small Sources)'
    ELSE primary_knowledge_source
  END as node_id,
  CASE
    WHEN total_from_primary < ${smallSourceThreshold} THEN 'Other (Small Sources)'
    ELSE display_name
  END as node_name,
  'primary' as category,
  0 as x_position,
  total_from_primary as value,
  NULL as total_all_sources
FROM primary_totals

UNION ALL

-- This KG node (aggregator)
SELECT
  ut.clean_upstream_source as node_id,
  ut.clean_upstream_source as node_name,
  'aggregator' as category,
  1 as x_position,
  ut.total_from_upstream as value,
  aut.total_all_sources as total_all_sources
FROM upstream_totals ut
LEFT JOIN all_upstream_totals aut ON aut.clean_upstream_source = ut.clean_upstream_source
```

```sql flow_links
-- Get data for network visualization - LINKS
WITH base_data AS (
  SELECT
    primary_knowledge_source,
    TRIM(upstream_source) as clean_upstream_source,
    SUM(count) as edge_count
  FROM bq.merged_kg_edges
  CROSS JOIN UNNEST(SPLIT(upstream_data_source, ',')) as t(upstream_source)
  WHERE TRIM(upstream_source) = '${params.knowledge_graph}'
    AND TRIM(upstream_source) IS NOT NULL
    AND TRIM(upstream_source) != ''
  GROUP BY 1, 2
),

primary_totals AS (
  SELECT
    base_data.primary_knowledge_source,
    SUM(edge_count) as total_from_primary
  FROM base_data
  GROUP BY base_data.primary_knowledge_source
)

-- Links from primary sources to this KG
SELECT
  edge_count as value,
  CASE
    WHEN (SELECT total_from_primary FROM primary_totals pt WHERE pt.primary_knowledge_source = bd.primary_knowledge_source) < ${smallSourceThreshold} THEN 'Other (Small Sources)'
    ELSE primary_knowledge_source
  END as source,
  clean_upstream_source as target
FROM base_data bd
```

{#if flow_nodes.length > 0}
<KnowledgeSourceFlowGraph
  nodeData={flow_nodes}
  linkData={flow_links}
  title="Knowledge Source Flow"
  topNPrimarySources={TOP_N_PRIMARY_SOURCES}
  height="700px"
  levelConfig={TWO_LEVEL_CONFIG}
/>
{:else}
<p class="text-gray-500">No flow data available for this knowledge graph.</p>
{/if}

## Normalization Pipeline

The normalization pipeline shows how data flows from the original knowledge graph through transformation and normalization stages.

```sql node_pipeline
select stage, count, sort_order
from bq.kg_pipeline_metrics
where knowledge_graph = '${params.knowledge_graph}' and entity_type = 'nodes'
order by sort_order
```

```sql edge_pipeline
select stage, count, sort_order
from bq.kg_pipeline_metrics
where knowledge_graph = '${params.knowledge_graph}' and entity_type = 'edges'
order by sort_order
```

<Grid cols=2>
<div>

### Nodes

{#if node_pipeline.length > 0}
<FunnelChart
  data={node_pipeline}
  nameCol="stage"
  valueCol="count"
  showPercent=true
/>
{:else}
<p class="text-gray-500">No node pipeline data available.</p>
{/if}

</div>
<div>

### Edges

{#if edge_pipeline.length > 0}
<FunnelChart
  data={edge_pipeline}
  nameCol="stage"
  valueCol="count"
  showPercent=true
/>
{:else}
<p class="text-gray-500">No edge pipeline data available.</p>
{/if}

</div>
</Grid>

## Epistemic Robustness

```sql kg_epistemic_score
WITH scored_edges AS (
  SELECT
    kl_score,
    at_score,
    kl_normalized,
    at_normalized,
    knowledge_level_label,
    agent_type_label,
    edge_count
  FROM bq.epistemic_scores
  WHERE upstream_data_source = '${params.knowledge_graph}'
),

most_common_kl AS (
  SELECT knowledge_level_label AS most_common_knowledge_level
  FROM scored_edges
  GROUP BY knowledge_level_label
  ORDER BY SUM(edge_count) DESC
  LIMIT 1
),

most_common_at AS (
  SELECT agent_type_label AS most_common_agent_type
  FROM scored_edges
  GROUP BY agent_type_label
  ORDER BY SUM(edge_count) DESC
  LIMIT 1
)

SELECT
  SUM(edge_count) AS included_edges,
  ROUND(
    SUM((kl_score + at_score) / 2 * edge_count) / SUM(edge_count),
    4
  ) AS average_epistemic_score,
  ROUND(SUM(kl_score * edge_count) / SUM(edge_count), 4) AS average_knowledge_level_score,
  ROUND(SUM(at_score * edge_count) / SUM(edge_count), 4) AS average_agent_type_score,
  SUM(CASE
    WHEN kl_normalized = 'not_provided' AND at_normalized = 'not_provided'
    THEN edge_count ELSE 0
  END) AS null_or_not_provided_both,
  (SELECT most_common_knowledge_level FROM most_common_kl) AS most_common_knowledge_level,
  (SELECT most_common_agent_type FROM most_common_at) AS most_common_agent_type
FROM scored_edges
```

<div class="text-left text-md max-w-3xl mx-auto mb-4">
  Epistemic Robustness measures the provenance quality of edges from this knowledge graph, using Knowledge Level and Agent Type from the Biolink Model to assess evidence strength and reliability. Higher scores indicate stronger evidence and greater manual curation.
</div>

{#if kg_epistemic_score.length > 0 && kg_epistemic_score[0].included_edges > 0}
<div class="text-center mb-6">
  <span class="font-semibold text-4xl" style="color: #9D79D6;">
    <Value data={kg_epistemic_score} column="average_epistemic_score" fmt="num2" />
  </span><br/>
  <span class="text-xl">Average Epistemic Score</span>
</div>

<Grid cols=2 class="max-w-4xl mx-auto mb-8">
  <div class="text-center text-lg">
    <span class="font-semibold text-3xl" style="color: #7C3AED;">
      <Value data={kg_epistemic_score} column="average_knowledge_level_score" fmt="num2" />
    </span><br/>
    <span class="text-lg">Knowledge Level Score</span><br/>
    <span class="text-sm text-gray-600 mt-1">
      Most common: <span class="font-semibold">{kg_epistemic_score[0].most_common_knowledge_level}</span>
    </span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-3xl" style="color: #7C3AED;">
      <Value data={kg_epistemic_score} column="average_agent_type_score" fmt="num2" />
    </span><br/>
    <span class="text-lg">Agent Type Score</span><br/>
    <span class="text-sm text-gray-600 mt-1">
      Most common: <span class="font-semibold">{kg_epistemic_score[0].most_common_agent_type}</span>
    </span>
  </div>
</Grid>

<Grid cols=2 class="max-w-4xl mx-auto mb-8">
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={kg_epistemic_score} column="included_edges" fmt="num0" />
    </span><br/>
    <span class="text-lg">Edges Included</span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={kg_epistemic_score} column="null_or_not_provided_both" fmt="num0" />
    </span><br/>
    <span class="text-lg">Edges with Missing Provenance</span><br/>
    <span class="text-xs text-gray-600">(Both Knowledge Level and Agent Type not provided)</span>
  </div>
</Grid>
{:else}
<div class="text-center text-lg text-gray-500 mb-6">
  No epistemic score data available for this knowledge graph.
</div>
{/if}

## Edge Type Validation

```sql kg_edge_validation_totals
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
  WHERE '${params.knowledge_graph}' IN (SELECT TRIM(s) FROM UNNEST(SPLIT(edges.upstream_data_source, ',')) AS s)
),
totals AS (
  SELECT
    validation_status,
    SUM(edge_count) AS total_edges
  FROM edge_validation
  GROUP BY validation_status
)
SELECT
  validation_status,
  total_edges,
  ROUND(100.0 * total_edges / SUM(total_edges) OVER (), 1) AS percentage
FROM totals
ORDER BY validation_status DESC
```

<div class="text-left text-md max-w-3xl mx-auto mb-6">
  This section shows the validation of edge types from this knowledge graph against the recognized biolink schema patterns.
</div>

{#if kg_edge_validation_totals.length > 0}
<Grid cols=2 class="max-w-4xl mx-auto mb-6">
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #1e40af;">
      {#if kg_edge_validation_totals.filter(d => d.validation_status === 'Recognized').length > 0}
        <Value data={kg_edge_validation_totals.filter(d => d.validation_status === 'Recognized')} column="total_edges" fmt="num0" />
      {:else}
        0
      {/if}
    </span><br/>
    <span class="text-xl">Recognized Edges</span><br/>
    <span class="text-lg" style="color: #1e40af;">
      {#if kg_edge_validation_totals.filter(d => d.validation_status === 'Recognized').length > 0}
        (<Value data={kg_edge_validation_totals.filter(d => d.validation_status === 'Recognized')} column="percentage" fmt="num1" />%)
      {:else}
        (0.0%)
      {/if}
    </span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #93c5fd;">
      {#if kg_edge_validation_totals.filter(d => d.validation_status === 'Unrecognized').length > 0}
        <Value data={kg_edge_validation_totals.filter(d => d.validation_status === 'Unrecognized')} column="total_edges" fmt="num0" />
      {:else}
        0
      {/if}
    </span><br/>
    <span class="text-xl">Unrecognized Edges</span><br/>
    <span class="text-lg" style="color: #93c5fd;">
      {#if kg_edge_validation_totals.filter(d => d.validation_status === 'Unrecognized').length > 0}
        (<Value data={kg_edge_validation_totals.filter(d => d.validation_status === 'Unrecognized')} column="percentage" fmt="num1" />%)
      {:else}
        (0.0%)
      {/if}
    </span>
  </div>
</Grid>
{:else}
<div class="text-center text-lg text-gray-500 mb-6">
  No edge validation data available for this knowledge graph.
</div>
{/if}

## ABox / TBox Classification

```sql kg_abox_tbox
WITH kg_edges AS (
  SELECT
    subject_category,
    object_category,
    SUM(count) as edge_count
  FROM bq.merged_kg_edges
  WHERE '${params.knowledge_graph}' IN (SELECT TRIM(s) FROM UNNEST(SPLIT(upstream_data_source, ',')) AS s)
  GROUP BY subject_category, object_category
),
classified AS (
  SELECT
    edge_count,
    CASE
      WHEN subject_category IN ('biolink:Gene', 'biolink:Protein', 'biolink:SmallMolecule', 'biolink:Drug', 'biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:ChemicalEntity', 'biolink:MolecularEntity', 'biolink:Pathway', 'biolink:BiologicalProcess', 'biolink:CellularComponent', 'biolink:Cell', 'biolink:AnatomicalEntity', 'biolink:GrossAnatomicalStructure')
           AND object_category IN ('biolink:Gene', 'biolink:Protein', 'biolink:SmallMolecule', 'biolink:Drug', 'biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:ChemicalEntity', 'biolink:MolecularEntity', 'biolink:Pathway', 'biolink:BiologicalProcess', 'biolink:CellularComponent', 'biolink:Cell', 'biolink:AnatomicalEntity', 'biolink:GrossAnatomicalStructure')
      THEN 'abox'
      WHEN subject_category LIKE '%Class%' OR subject_category LIKE '%Ontology%' OR object_category LIKE '%Class%' OR object_category LIKE '%Ontology%'
      THEN 'tbox'
      ELSE 'undefined'
    END as classification
  FROM kg_edges
)
SELECT
  SUM(CASE WHEN classification = 'abox' THEN edge_count ELSE 0 END) as abox,
  SUM(CASE WHEN classification = 'tbox' THEN edge_count ELSE 0 END) as tbox,
  SUM(CASE WHEN classification = 'undefined' THEN edge_count ELSE 0 END) as undefined,
  SUM(edge_count) as total_edges,
  ROUND(100.0 * SUM(CASE WHEN classification = 'abox' THEN edge_count ELSE 0 END) / NULLIF(SUM(edge_count), 0), 1) as abox_percentage,
  ROUND(100.0 * SUM(CASE WHEN classification = 'tbox' THEN edge_count ELSE 0 END) / NULLIF(SUM(edge_count), 0), 1) as tbox_percentage,
  ROUND(100.0 * SUM(CASE WHEN classification = 'undefined' THEN edge_count ELSE 0 END) / NULLIF(SUM(edge_count), 0), 1) as undefined_percentage
FROM classified
```

The TBox-to-ABox balance reflects how much a knowledge graph emphasizes abstract schema versus concrete instances—too much of either can hinder effective learning and reasoning.

<Details title="About This Metric">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
ABox (Assertional Box) edges represent instance-level relationships between specific entities, while TBox
(Terminological Box) edges represent concept-level relationships between types or classes.
This classification helps understand the semantic nature of the knowledge graph.
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
A high TBox ratio suggests the graph contains mostly general, ontological structure, which may add noise and limit the
discovery of meaningful, specific patterns by machine learning models.
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
Conversely, a low TBox ratio might mean the graph is missing useful schema-level structure that could aid clustering or
reasoning - e.g., similar nodes may not be grouped efficiently in embedding space.
</div>
</Details>

{#if kg_abox_tbox.length > 0 && kg_abox_tbox[0].total_edges > 0}
<Grid cols=2 class="max-w-4xl mx-auto mb-6">
  <div>
    <ECharts config={{
        title: {
            text: 'Classification',
            left: 'center',
            top: 'center',
            textStyle: {
                fontWeight: 'normal'
            }
        },
        color: ['#6287D3', '#D8AB47', '#AAAAAA'],
        tooltip: {
            formatter: function(params) {
                const count = params.data.value.toLocaleString();
                return params.name + ': ' + count + ' edges (' + params.percent + '%)';
            }
        },
        series: [{
            type: 'pie',
            data: [
              { value: kg_abox_tbox[0].abox, name: 'ABox (Instance-level)' },
              { value: kg_abox_tbox[0].tbox, name: 'TBox (Concept-level)' },
              { value: kg_abox_tbox[0].undefined, name: 'Undefined' }
            ],
            radius: ['40%', '65%']
        }]
    }}/>
  </div>
  <div class="text-center flex flex-col justify-center">
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: #6287D3;">
        <Value data={kg_abox_tbox} column="abox" fmt="num0" />
      </span><br/>
      <span class="text-base">ABox Edges</span><br/>
      <span class="text-xs text-gray-600">
        (<Value data={kg_abox_tbox} column="abox_percentage" fmt="num1" />%)
      </span>
    </div>
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: #D8AB47;">
        <Value data={kg_abox_tbox} column="tbox" fmt="num0" />
      </span><br/>
      <span class="text-base">TBox Edges</span><br/>
      <span class="text-xs text-gray-600">
        (<Value data={kg_abox_tbox} column="tbox_percentage" fmt="num1" />%)
      </span>
    </div>
    <div class="mb-3">
      <span class="font-semibold text-2xl" style="color: #AAAAAA;">
        <Value data={kg_abox_tbox} column="undefined" fmt="num0" />
      </span><br/>
      <span class="text-base">Undefined</span><br/>
      <span class="text-xs text-gray-600">
        (<Value data={kg_abox_tbox} column="undefined_percentage" fmt="num1" />%)
      </span>
    </div>
  </div>
</Grid>
{:else}
<div class="text-center text-lg text-gray-500 mb-6">
  No ABox/TBox data available for this knowledge graph.
</div>
{/if}

## Contribution Summary

This section shows the top categories and predicates contributed by this knowledge graph to the unified graph.

<Grid cols=2>
<div>

### Top Category Pairs

{#if top_categories.length > 0}
<BarChart
  data={top_categories}
  x="category_pair"
  y="edge_count"
  swapXY=true
  fmt="num0"
  title="Top 10 Category Pairs by Edge Count"
/>
{:else}
<p class="text-gray-500">No category data available.</p>
{/if}

</div>
<div>

### Top Predicates

{#if top_predicates.length > 0}
<BarChart
  data={top_predicates}
  x="predicate_display"
  y="edge_count"
  swapXY=true
  fmt="num0"
  title="Top 10 Predicates by Edge Count"
/>
{:else}
<p class="text-gray-500">No predicate data available.</p>
{/if}

</div>
</Grid>

{/if}
