# {params.upstream}

{params.upstream} Normalization Failures

```sql edge_failed_normalization
select original_prefix, 
       normalization_set,
       '/normalization/failures/' || normalization_set  || '/' || prefix as link,
       sum(count) as count
from bq.normalization
where normalization_success = false
  and normalization_set = '${params.upstream}'
group by all
order by count desc
```

Normalization failures are counted when a subject or object column is marked as having a failed 
normalization. For each prefix, examples of failed normalization are captured and can be accessed by clicking on the Examples link. 


<DataTable data={edge_failed_normalization}
  title="Normalization Failures"
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