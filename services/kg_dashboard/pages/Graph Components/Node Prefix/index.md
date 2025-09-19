
```sql nodes_by_prefix
    select 
        prefix,
        '/Graph Components/Node Prefix/' || prefix as link,
        sum(count) as count
    from bq.merged_kg_nodes
    group by all
    order by count desc
```
Click through to node prefixes in the table below to explore more detailed information about each of them.

<DataTable 
    data={nodes_by_prefix} 
    search=true
    link=link 
    title='Nodes by Prefix' />
