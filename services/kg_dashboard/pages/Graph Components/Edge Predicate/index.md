```sql predicates_by_count
select 
    replace(predicate,'biolink:','') as predicate,
    '/Graph Components/Edge Predicate/' || replace(predicate,'biolink:','') as link,
    sum(count) as count
from bq.merged_kg_edges
group by all
order by count desc  
```
Click through to predicates in the table below to explore more detailed information about each of them.

<DataTable 
  data={predicates_by_count} 
  search=true
  link=link 
  title='Predicates by Edge Count' />