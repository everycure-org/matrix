# {params.prefix}
# {params.upstream}

{params.upstream} {params.prefix} Normalization Failure Examples


```sql edge_failed_normalization_examples
select 
  id, 
  name,
  category, 
  normalization_set as upstream_data_source,
  'https://bioregistry.io/' || id as link,
from bq.edge_failed_normalization_examples
where prefix = '${params.prefix}'
  and normalization_set = '${params.upstream}'
```

```sql normalization_sets
select normalization_set, count(*)
from bq.edge_failed_normalization_examples
group by all
```

These are examples of {params.prefix} identifiers from {params.upstream} that failed normalization, the > link will use bioregistry.io link expansion to provide more information about the idenfifier.

{#if edge_failed_normalization_examples.length > 0}

<DataTable 
    title="Failed Normalization Examples"
    data={edge_failed_normalization_examples} 
    search=true
    pagination=true
    link=link
    rows=25
/>

{:else}

No failed normalization examples found for {params.prefix} from {params.upstream}.

{/if}
