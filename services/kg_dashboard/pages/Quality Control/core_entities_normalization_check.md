---
title: Core entities normalization check
---

```sql core_entities_normalization_check
SELECT * FROM bq.core_entities_normalization_check
```

Drug and diseases come from our core-entities repository, and should not be normalized again, otherwise we would end up with integration problems between our systems. Please find below the nodes that were normalized again.


<div class="text-center text-lg mt-6 mb-6">
    {#if core_entities_normalization_check.length > 0}

    <p class="text-red-500 font-bold mb-2">{core_entities_normalization_check.length} nodes were normalized again</p>
    <p class="text-red-500">You can find them in the table below</p>

    {:else}

    <p class="text-lg text-green-500">All good, no nodes were normalized again</p>

    {/if}
</div>

<DataTable
  data={core_entities_normalization_check}
  columns={['source', 'original_id', 'id', 'name', 'category']}
  pagination
/>
