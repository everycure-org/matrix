# {params.upstream}

{params.upstream} Unrecognized Normalization

```sql edge_failed_normalization
select original_prefix, 
       normalization_set,
       '/normalization/unrecognized/' || normalization_set  || '/' || prefix as link,
       sum(count) as count
from bq.normalization
where normalization_success = false
  and normalization_set = '${params.upstream}'
group by all
order by count desc
```

Unrecognized normalizations are counted when a subject or object column is marked with normalization_success `false`. For each data source,examples of failed normalization are captured and can be accessed by clicking on the Examples link. 

<DataTable data={edge_failed_normalization}
  title="Unrecognized Normalization"
  search=true
  pagination=true
  link=true
  rows=20
>
  <Column id="normalization_set" title="Upstream Data Source" />
  <Column id="original_prefix" />
  <Column id="count" contentType="bar"/>
  <Column id="link" contentType="link" linkLabel="Examples ->" title=" " />
</DataTable>