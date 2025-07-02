---
title: EC Core Components
---

This page provides the count of EC Core Components from the disease list and drug list, compared with the unified nodes table.

# Disease List:
```sql disease_list_normalization
SELECT * FROM bq.ec_core_components_disease_list
``` 

```sql disease_list_success
SELECT * FROM bq.ec_core_components_disease_list_success
``` 

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
            <DataTable 
                data={disease_list_success 
                    .filter(d => d.normalization_success === true && d.dimension === 'category') 
                    .map(({ name, count }) => ({ name, count }))} 
                columns={['name', 'count']} 
                pagination 
                pageSize={10} 
            /> 
        </Tab> 
        <Tab label="Failure">
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


# Drug List:
```sql drug_list_normalization
SELECT * FROM bq.ec_core_components_drug_list
```

```sql drug_list_success
SELECT * FROM bq.ec_core_components_drug_list_success
```

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
            <DataTable 
                data={drug_list_success 
                    .filter(d => d.normalization_success === true && d.dimension === 'category') 
                    .map(({ name, count }) => ({ name, count }))}  
                columns={['name', 'count']} 
                pagination 
                pageSize={10} 
            /> 
        </Tab> 
        <Tab label="Failure"> 
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
