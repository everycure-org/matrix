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

{:else}
<div class="text-center text-lg text-red-500 mt-10">
  Knowledge source "{params.knowledge_source}" not found in infores catalog.
</div>
{/if}

