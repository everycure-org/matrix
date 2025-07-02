
# Drug List

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

## Drug List (Missing):

The data table contains all the entities that are present in the drug_list_nodes_normalized table, but are missing from the nodes_unified table.

```sql drug_list_missing
SELECT * FROM bq.ec_core_components_drug_list_missing
```

<DataTable
    data={drug_list_missing}
    columns={['id', 'name']}
    pagination
    pageSize={10}
/>
