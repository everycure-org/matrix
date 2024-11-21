# {params.prefix}

Title: {params.prefix} Dashboard


```sql node_categories_by_upstream_data_source
select category, upstream_data_source, sum(count) as count 
from bq.merged_kg_nodes
where prefix = '${params.prefix}'
group by all
having count > 0
order by count desc
limit 50
```


```sql edge_types_by_upstream_data_source
  select 
      replace(subject_category,'biolink:','') || ' ' ||
      replace(predicate,'biolink:','') || ' ' || 
      replace(object_category,'biolink:','') as edge_type,
      upstream_data_source,
      sum(count) as count
  from bq.merged_kg_edges
  where subject_prefix = '${params.prefix}'
    or object_prefix = '${params.prefix}'    
  group by all
  having count > 0
  order by count desc
```  

```sql edge_counts_by_primary_knowledge_source
  select
      primary_knowledge_source,
      upstream_data_source,
      sum(count) as count
  from bq.merged_kg_edges
  where subject_prefix = '${params.prefix}'
    or object_prefix = '${params.prefix}'
  group by all
  having count > 0
  order by count desc
```

{#if node_categories_by_upstream_data_source.length !== 0}
<BarChart 
    data={node_categories_by_upstream_data_source}
    x=category
    y=count
    series=upstream_data_source
    swapXY=true    
    title="Node Categories by Upstream Data Source"
/>
{/if}

{#if edge_types_by_upstream_data_source.length !== 0}
<BarChart
    data={edge_types_by_upstream_data_source}
    x=edge_type
    y=count
    split=upstream_data_source
    swapXY=true
    />
{/if}

{#if edge_counts_by_primary_knowledge_source.length !== 0}
<BarChart
    data={edge_counts_by_primary_knowledge_source}
    x=primary_knowledge_source
    y=count
    split=upstream_data_source
    title="Edge Counts by Primary Knowledge Source"
    swapXY=true
/>
{/if}
