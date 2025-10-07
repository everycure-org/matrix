# {params.predicate}

<p class="text-sm mb-4">
  <a class="underline text-blue-600" href="https://biolink.github.io/biolink-model/{params.predicate}/" target="_blank">View biolink model definition →</a>
</p>

<script context="module">
  import { getSeriesColors, sourceOrder, sortDataBySeriesOrder } from '../../../_lib/colors';
  
  // Use the shared sorting function from colors.js
  export const sortBySeriesOrdered = sortDataBySeriesOrder;
</script>

```sql number_of_edges
select coalesce(sum(count), 0) as count
from bq.merged_kg_edges
where predicate = 'biolink:${params.predicate}'
```

```sql three_level_sankey
-- 3-level sankey: subject_category -> predicate -> object_category
SELECT 
    concat('[S] ', replace(subject_category,'biolink:','')) as source,
    concat('[P] ', replace(predicate,'biolink:','')) as target,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE predicate = 'biolink:${params.predicate}'
GROUP BY all

UNION ALL

SELECT 
    concat('[P] ', replace(predicate,'biolink:','')) as source,
    concat('[O] ', replace(object_category,'biolink:','')) as target,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE predicate = 'biolink:${params.predicate}'
GROUP BY all

ORDER BY count DESC
```


```sql primary_knowledge_source_counts
select
    primary_knowledge_source,
    sum(count) as count
from bq.merged_kg_edges
where predicate = 'biolink:${params.predicate}'
group by all
having count > 0
order by count desc
```


<p class="text-center text-lg py-4"><span class="font-semibold text-2xl"><Value data={number_of_edges} column="count" fmt="integer"/></span><br/>edges using this predicate</p>


<SankeyDiagram 
    data={three_level_sankey}
    sourceCol='source'
    targetCol='target'
    valueCol='count'
    linkLabels='full'
    linkColor='gradient'
    chartAreaHeight={600}
    title="Subject Category → {params.predicate} → Object Category Flow"
/>

<BarChart
    data={primary_knowledge_source_counts}
    x=primary_knowledge_source
    y=count
    title="Edge Counts by Primary Knowledge Source"
/>


