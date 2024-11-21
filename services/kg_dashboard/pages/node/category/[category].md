# {params.category}

Title: {params.category} Dashboard


```sql node_prefixes_by_upstream_data_source
select prefix, upstream_data_source, sum(count) as count 
from bq.merged_kg_nodes
where category = 'biolink:${params.category}'
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
  where subject_category = 'biolink:${params.category}'
    or object_category = 'biolink:${params.category}'    
  group by all
  having count > 0
  order by count desc
```  

{#if node_prefixes_by_upstream_data_source.length !== 0}
<BarChart 
    data={node_prefixes_by_upstream_data_source}
    x=prefix
    y=count
    series=upstream_data_source
    swapXY=true    
    title="Node Prefixes by Upstream Data Source"
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
