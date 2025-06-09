---
title: Core entities
---
<script>
  const max_core_entities_normalization_errors = import.meta.env.VITE_max_core_entities_normalization_errors;
</script>

# Summary

<Details title="Core entities' ids should not change during normalization, click here to understand why">

  We normalize core entities' ids in the MATRIX pipeline to make sure we can join them with our other data sources and Knowledge Graphs. However, their id should not change during normalization, otherwise we will end up with id conflict in downstream systems like ORCHARD as they consume both core-entities and MATRIX outputs.

  This is an iteration on this issue and is likely to change in the future.
</Details>

```sql core_entities_re_normalization_errors
select 
  source, original_id, id, category, name 
from
  bq.core_entities_normalization_errors 
where 
  id != original_id
```

```sql core_entities_normalization_failure_errors
select 
  source, original_id, id, category, name 
from
  bq.core_entities_normalization_errors 
where 
  not normalization_success
```

<div class="text-center text-lg mt-6 mb-6 space-y-4">
    {#if core_entities_re_normalization_errors.length > 0 || core_entities_normalization_failure_errors.length > 0}

      <p class="text-red-500">❌ Some errors require your attention</p>
      
      {#if core_entities_re_normalization_errors.length > 0}
        {#if core_entities_re_normalization_errors.length === max_core_entities_normalization_errors}
          <p class="font-bold">At least {max_core_entities_normalization_errors} IDs changed during normalization</p>
        {:else}
          <p class="font-bold">{core_entities_re_normalization_errors.length} IDs changed during normalization</p>
        {/if}
        <DataTable
          data={core_entities_re_normalization_errors}
          columns={['source', 'original_id', 'id', 'category', 'name']}
          pagination
        />
      {/if}

      {#if core_entities_normalization_failure_errors.length > 0}
        {#if core_entities_normalization_failure_errors.length === max_core_entities_normalization_errors}
          <p class="font-bold">At least {max_core_entities_normalization_errors} nodes were not normalized</p>
        {:else}
          <p class="font-bold">{core_entities_normalization_failure_errors.length} nodes were not normalized</p>
        {/if}
        <DataTable
          data={core_entities_normalization_failure_errors}
          columns={['source', 'original_id', 'id', 'category', 'name']}
            pagination
          />
      {/if}

    {:else}
      <p class="text-lg text-green-500">✅ All good</p>
    {/if}
</div>

<div class="h-4"/>

```sql disease_list_normalization
SELECT * FROM bq.disease_list_normalization
```

```sql disease_list_success
SELECT * FROM bq.disease_list_success
```

# Disease list

<Grid col=2>
    <ECharts
        config={{
            tooltip: {
                formatter: function(params) {
                    const count = params.data.value.toLocaleString();
                    return `${params.name}: ${count} nodes (${params.percent}%)`;
                }
            },
            series: [{
                type: 'pie', 
                radius: ['30%', '50%'],
                data: disease_list_normalization.map(d => ({
                    ...d,
                    itemStyle: {
                      color: d.name === 'Success' ? '#50fa7b' : '#ff5555' 
                    }
                }))
            }]
        }}
    />
    <Tabs fullWidth=true>
        <Tab label="Success">
            <Tabs>
              <Tab label="Prefixes">
                <DataTable
                          data={disease_list_success
                            .filter(d => d.normalization_success === true && d.dimension === 'prefix')
                            .map(({ name, count }) => ({ name, count }))}
                          columns={['name', 'count']}
                          pagination
                          pageSize={10}
                        />
              </Tab>
              <Tab label="Categories">
                <DataTable
                          data={disease_list_success
                            .filter(d => d.normalization_success === true && d.dimension === 'category')
                            .map(({ name, count }) => ({ name, count }))}
                          columns={['name', 'count']}
                          pagination
                          pageSize={10}
                        />
              </Tab>
            </Tabs>
        </Tab>
        <Tab label="Failure">
            <Tabs>
              <Tab label="Prefixes">
                <DataTable
                          data={disease_list_success
                            .filter(d => d.normalization_success === false && d.dimension === 'prefix')
                            .map(({ name, count }) => ({ name, count }))}
                          columns={['name', 'count']}
                          pagination
                          pageSize={10}
                        />
              </Tab>
              <Tab label="Categories">
                <DataTable
                          data={disease_list_success
                            .filter(d => d.normalization_success === false && d.dimension === 'category')
                            .map(({ name, count }) => ({ name, count }))}
                          columns={['name', 'count']}
                          pagination
                          pageSize={10}
                        />
              </Tab>
            </Tabs>
        </Tab>
    </Tabs>
</Grid>

```sql drug_list_normalization
SELECT * FROM bq.drug_list_normalization
```

```sql drug_list_success
SELECT * FROM bq.drug_list_success
```

# Drug list 

<Grid col=2>
    <ECharts
        config={{
            tooltip: {
                formatter: function(params) {
                    const count = params.data.value.toLocaleString();
                    return `${params.name}: ${count} nodes (${params.percent}%)`;
                }
            },
            series: [{
                type: 'pie', 
                radius: ['30%', '50%'],
                data: drug_list_normalization.map(d => ({
                    ...d,
                    itemStyle: {
                      color: d.name === 'Success' ? '#50fa7b' : '#ff5555' 
                    }
                }))
            }]
        }}
    />
    <Tabs fullWidth=true>
        <Tab label="Success">
            <Tabs>
              <Tab label="Prefixes">
                <DataTable
                          data={drug_list_success
                            .filter(d => d.normalization_success === true && d.dimension === 'prefix')
                            .map(({ name, count }) => ({ name, count }))}
                          columns={['name', 'count']}
                          pagination
                          pageSize={10}
                        />
              </Tab>
              <Tab label="Categories">
                <DataTable
                          data={drug_list_success
                            .filter(d => d.normalization_success === true && d.dimension === 'category')
                            .map(({ name, count }) => ({ name, count }))}
                          columns={['name', 'count']}
                          pagination
                          pageSize={10}
                        />
              </Tab>
            </Tabs>
        </Tab>
        <Tab label="Failure">
            <Tabs>
              <Tab label="Prefixes">
                <DataTable
                          data={drug_list_success
                            .filter(d => d.normalization_success === false && d.dimension === 'prefix')
                            .map(({ name, count }) => ({ name, count }))}
                          columns={['name', 'count']}
                          pagination
                          pageSize={10}
                        />
              </Tab>
              <Tab label="Categories">
                <DataTable
                          data={drug_list_success
                            .filter(d => d.normalization_success === false && d.dimension === 'category')
                            .map(({ name, count }) => ({ name, count }))}
                          columns={['name', 'count']}
                          pagination
                          pageSize={10}
                        />
              </Tab>
            </Tabs>
        </Tab>
    </Tabs>
</Grid>