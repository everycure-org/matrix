
# Disease List

```sql disease_list_normalization
SELECT * FROM bq.ec_core_components_disease_list
``` 

```sql disease_list_success
SELECT * FROM bq.ec_core_components_disease_list_success
``` 

<div class="h-4"/>

```sql disease_list_normalization
SELECT * FROM bq.disease_list_normalization
```

```sql disease_list_success
SELECT * FROM bq.disease_list_success
```

# Disease list normalization

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

## Disease List (Missing):

The data table contains all the entities that are present in the disease_list_nodes_normalized table, but are missing from the nodes_unified table.

```sql disease_list_missing
SELECT * FROM bq.ec_core_components_disease_list_missing
```

<DataTable
    data={disease_list_missing}
    columns={['id', 'name']}
    pagination
    pageSize={10}
/>