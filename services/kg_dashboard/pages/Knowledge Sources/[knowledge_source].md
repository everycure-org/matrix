# {knowledge_source_info.length > 0 ? knowledge_source_info[0].name || params.knowledge_source : params.knowledge_source}

```sql knowledge_source_info
SELECT 
 id,
 name,
 description,
 xref, 
 synonym,
 knowledge_level,
 agent_type
FROM infores.catalog
WHERE id = '${params.knowledge_source}'
```


{#if knowledge_source_info.length > 0}

<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  <strong>Description:</strong> {knowledge_source_info[0].description || 'No description available'}
</div>
{#if knowledge_source_info[0].xref}
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  <strong>Cross-references:</strong> 
  {#each knowledge_source_info[0].xref.split(',') as xref, i}
    <!-- Using span with window.open() instead of <a> tag to bypass Evidence router and force external links -->
    <span 
      class="underline text-blue-600 cursor-pointer" 
      on:click={() => window.open(xref.trim(), '_blank')}
    >{xref.trim()}</span>{#if i < knowledge_source_info[0].xref.split(',').length - 1}, {/if}
  {/each}
</div>
{/if}
{#if knowledge_source_info[0].synonym}
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  <strong>Synonyms:</strong> {knowledge_source_info[0].synonym}
</div>
{/if}

## Edge Categories Distribution

```sql sankey_data
-- First level: Subject Category to Predicate
SELECT 
    concat('[S] ', replace(subject_category,'biolink:','')) as source,
    replace(predicate,'biolink:','') as target,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE primary_knowledge_source = '${params.knowledge_source}'
GROUP BY all

UNION ALL

-- Second level: Predicate to Object Category
SELECT 
    replace(predicate,'biolink:','') as source,
    concat('[O] ', replace(object_category,'biolink:','')) as target,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE primary_knowledge_source = '${params.knowledge_source}'
GROUP BY all
ORDER BY count DESC
```

{#if sankey_data.length > 0}
<SankeyDiagram data={sankey_data} 
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  title='Knowledge Source Edge Flow'
  subtitle='Flow from Subject Categories through Predicates to Object Categories'
  chartAreaHeight={800}
/>
{:else}
<div class="text-center text-lg text-gray-500 mt-10">
  No edge data found for knowledge source "{params.knowledge_source}".
</div>
{/if}

## Epistemic Robustness

```sql ks_epistemic_score
-- Uses centralized scoring from epistemic_scores source
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
  WHERE primary_knowledge_source = '${params.knowledge_source}'
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
  Epistemic Robustness measures the provenance quality of edges from this knowledge source, using Knowledge Level and Agent Type from the Biolink Model to assess evidence strength and reliability. Higher scores indicate stronger evidence and greater manual curation.
</div>

{#if ks_epistemic_score.length > 0}
<div class="text-center mb-6">
  <span class="font-semibold text-4xl" style="color: #9D79D6;">
    <Value data={ks_epistemic_score} column="average_epistemic_score" fmt="num2" />
  </span><br/>
  <span class="text-xl">Average Epistemic Score</span>
</div>

<Grid col=2 class="max-w-4xl mx-auto mb-15">
  <div class="text-center text-lg">
    <span class="font-semibold text-3xl" style="color: #7C3AED;">
      <Value data={ks_epistemic_score} column="average_knowledge_level_score" fmt="num2" />
    </span><br/>
    <span class="text-lg">Knowledge Level Score</span><br/>
    <span class="text-sm text-gray-600 mt-1">
      Most common: <span class="font-semibold">{ks_epistemic_score[0].most_common_knowledge_level}</span>
    </span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-3xl" style="color: #7C3AED;">
      <Value data={ks_epistemic_score} column="average_agent_type_score" fmt="num2" />
    </span><br/>
    <span class="text-lg">Agent Type Score</span><br/>
    <span class="text-sm text-gray-600 mt-1">
      Most common: <span class="font-semibold">{ks_epistemic_score[0].most_common_agent_type}</span>
    </span>
  </div>
</Grid>

## 

<Grid col=2 class="max-w-4xl mx-auto mb-8, mt-15">
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={ks_epistemic_score} column="included_edges" fmt="num0" />
    </span><br/>
    <span class="text-lg">Edges Included</span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-2xl">
      <Value data={ks_epistemic_score} column="null_or_not_provided_both" fmt="num0" />
    </span><br/>
    <span class="text-lg">Edges with Missing Provenance</span><br/>
    <span class="text-xs text-gray-600">(Both Knowledge Level and Agent Type not provided)</span>
  </div>
</Grid>
{:else}
<div class="text-center text-lg text-gray-500 mb-6">
  No epistemic score data available for this knowledge source.
</div>
{/if}

## Edge Type Validation

```sql ks_edge_validation
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
  WHERE edges.primary_knowledge_source = '${params.knowledge_source}'
),
totals AS (
  SELECT 
    validation_status,
    SUM(edge_count) AS total_edges
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
  'Edge Validation' as category,
  validation_status,
  percentage
FROM percentages

UNION ALL

-- Ensure both statuses are represented even if one has 0 edges
SELECT 
  'Edge Validation' as category,
  'Recognized' as validation_status,
  0.0 as percentage
WHERE NOT EXISTS (SELECT 1 FROM percentages WHERE validation_status = 'Recognized')

UNION ALL

SELECT 
  'Edge Validation' as category,
  'Unrecognized' as validation_status,
  0.0 as percentage
WHERE NOT EXISTS (SELECT 1 FROM percentages WHERE validation_status = 'Unrecognized')
```

```sql ks_edge_validation_totals
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
  WHERE edges.primary_knowledge_source = '${params.knowledge_source}'
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
  This section shows the validation of edge types in this knowledge source against the recognized biolink schema patterns.
</div>

{#if ks_edge_validation_totals.length > 0}
<Grid col=2 class="max-w-4xl mx-auto mb-6">
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #1e40af;">
      {#if ks_edge_validation_totals.filter(d => d.validation_status === 'Recognized').length > 0}
        <Value data={ks_edge_validation_totals.filter(d => d.validation_status === 'Recognized')} column="total_edges" fmt="num0" />
      {:else}
        0
      {/if}
    </span><br/>
    <span class="text-xl">Recognized Edges</span><br/>
    <span class="text-lg" style="color: #1e40af;">
      {#if ks_edge_validation_totals.filter(d => d.validation_status === 'Recognized').length > 0}
        (<Value data={ks_edge_validation_totals.filter(d => d.validation_status === 'Recognized')} column="percentage" fmt="num1" />%)
      {:else}
        (0.0%)
      {/if}
    </span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #93c5fd;">
      {#if ks_edge_validation_totals.filter(d => d.validation_status === 'Unrecognized').length > 0}
        <Value data={ks_edge_validation_totals.filter(d => d.validation_status === 'Unrecognized')} column="total_edges" fmt="num0" />
      {:else}
        0
      {/if}
    </span><br/>
    <span class="text-xl">Unrecognized Edges</span><br/>
    <span class="text-lg" style="color: #93c5fd;">
      {#if ks_edge_validation_totals.filter(d => d.validation_status === 'Unrecognized').length > 0}
        (<Value data={ks_edge_validation_totals.filter(d => d.validation_status === 'Unrecognized')} column="percentage" fmt="num1" />%)
      {:else}
        (0.0%)
      {/if}
    </span>
  </div>
</Grid>
{/if}


{:else}
<div class="text-center text-lg text-red-500 mt-10">
  Knowledge source "{params.knowledge_source}" not found in infores catalog.
</div>
{/if}

