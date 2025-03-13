# {params.prefix}

{params.prefix} Normalization Failure Examples

```sql edge_failed_normalization_examples
select 
  id, 
  name,
  category, 
  normalization_set as upstream_data_source,
  'https://bioregistry.io/' || id as link,
from bq.edge_failed_normalization_examples
where prefix = '${params.prefix}'
```

These are examples of {params.prefix} identifiers that failed normalization, the > link will use bioregistry.io link expansion to provide more information about the idenfifier.

<DataTable 
    title="Failed Normalization Examples"
    data={edge_failed_normalization_examples} 
    search=true
    pagination=true
    link=link
    rows=25
/>
