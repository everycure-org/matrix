--
Explore Graph Nodes
--

```sql nodes_by_category
  select 
      category,
      '/node/category/' || replace(category,'biolink:','') as link,
      sum(count) as count
  from bq.merged_kg_nodes
  group by all
  order by count desc  
```

```sql nodes_by_prefix
    select 
        prefix,
        '/node/prefix/' || prefix as link,
        sum(count) as count
    from bq.merged_kg_nodes
    group by all
    order by count desc
```

<DataTable data={nodes_by_category} link=link title='Nodes by Category' />
<DataTable data={nodes_by_prefix} link=link title='Nodes by Prefix' />
