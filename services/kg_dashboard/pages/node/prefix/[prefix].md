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

<BarChart 
    data={node_categories_by_upstream_data_source}
    x=category
    y=count
    series=upstream_data_source
    swapXY=true    
    title="Node Categories by Upstream Data Source"
/>

<BarChart
    data={edge_types_by_upstream_data_source}
    x=edge_type
    y=count
    split=upstream_data_source
    swapXY=true
    />
