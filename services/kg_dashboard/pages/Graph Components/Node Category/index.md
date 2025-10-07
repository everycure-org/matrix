```sql nodes_by_category
  select 
      category,
      '/Graph Components/Node Category/' || replace(category,'biolink:','') as link,
      sum(count) as count
  from bq.merged_kg_nodes
  group by all
  order by count desc  
```
Click through to node categories in the table below to explore more detailed information about each of them.


<DataTable 
  data={nodes_by_category} 
  search=true
  link=link 
  title='Nodes by Category' />
