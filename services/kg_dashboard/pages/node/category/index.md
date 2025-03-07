```sql nodes_by_category
  select 
      category,
      '/node/category/' || replace(category,'biolink:','') as link,
      sum(count) as count
  from bq.merged_kg_nodes
  group by all
  order by count desc  
```

<DataTable data={nodes_by_category} link=link title='Nodes by Category' />
