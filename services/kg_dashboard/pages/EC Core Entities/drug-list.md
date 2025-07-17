# Drug List

<script>
  const max_core_entities_normalization_errors = import.meta.env.VITE_max_core_entities_normalization_errors;

  function getCSSColor(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  const positiveColor = getCSSColor('--color-positive');
  const negativeColor = getCSSColor('--color-negative');
</script>

<Details title="Core entities' IDs should not change during normalization — click to learn why">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
Normalization is the process of mapping identifiers to standardized, canonical forms so they can be reliably joined across 
data sources. In the MATRIX pipeline, we use the <a href="https://github.com/TranslatorSRI/NodeNormalization" 
class="underline text-blue-600" target="_blank">Node Normalizer</a> to ensure consistency in how entities are represented. 
This tool resolves various identifiers and synonyms to preferred IDs, supporting harmonization across datasets.
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
However, for core entities, their IDs should not change during normalization. If they do, it can lead to conflicts in downstream 
systems like ORCHARD, which consume both the original core-entities and MATRIX outputs. 
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This is an active area of iteration and may evolve in future versions.
</div>
</Details>

```sql core_entities_re_normalization_errors
select 
  source, original_id, id, category, name 
from
  bq.core_entities_normalization_errors 
where 
  id != original_id
  AND source = 'drug_list'
```

```sql core_entities_normalization_failure_errors
select 
  source, original_id, id, category, name 
from
  bq.core_entities_normalization_errors 
where 
  not normalization_success
  AND source = 'drug_list'
```

<div class="text-center text-lg mt-6 mb-6 space-y-4">
    {#if core_entities_re_normalization_errors.length > 0 || core_entities_normalization_failure_errors.length > 0}

      <p style="color: {negativeColor}">❌ Some errors require your attention</p>
      
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
      <p style="color: {positiveColor}">✅ All good</p>
    {/if}
</div>

```sql drug_list_normalization
SELECT * FROM bq.drug_list_normalization
```

```sql drug_list_success
SELECT * FROM bq.drug_list_success
```

## Drug list normalization

<Details title="Details">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This section shows how successfully the entities in the drug list were normalized. 
The donut chart summarizes the proportion of nodes that were successfully mapped to standardized identifiers (Success) 
versus those that failed normalization (Failure). The accompanying table breaks down the successes and failures 
by prefix and category.
</div>
</Details>

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
                      color: d.name === 'Success' ? positiveColor : negativeColor
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

## Drug list inclusion

<Details title="Details">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This section summarizes how successfully entities from the drug list were integrated into the merged knowledge graph.
The donut chart shows the proportion of nodes that were included versus those that were missing.
The accompanying tables provide detailed lists of the nodes in each category.
</div>
</Details>

```sql disease_list_included_missing
SELECT * FROM bq.ec_core_components_drug_list_missing
```

<Grid col=2>
    <ECharts
        style={{ height: '500px' }},
        config={{
            tooltip: {
                formatter: function(params) {
                    const count = params.data.value.toLocaleString();
                    return `${params.name}: ${count} nodes (${params.percent}%)`;
                }
            },
            series: [{
                type: 'pie',
                center: ['50%', '50%'],
                radius: ['30%', '50%'],
                data: (() => {
                    const totals = disease_list_included_missing.reduce((acc, row) => {
                        acc[row.status] = (acc[row.status] || 0) + 1;
                        return acc;
                    }, {});
                    return Object.entries(totals).map(([status, count]) => ({
                        name: status,
                        value: count,
                        itemStyle: {
                            color: status === 'Included' ? positiveColor : negativeColor
                        }
                    }));
                })()
            }]
        }}
    />

    <Tabs fullWidth=true>
        <Tab label="Included">
            <DataTable
                data={disease_list_included_missing
                    .filter(d => d.status === 'Included')
                    .map(({ id, name }) => ({ id, name }))}
                columns={['id', 'name']}
                pagination
                pageSize={10}
            />
        </Tab>
        <Tab label="Missing">
            <DataTable
                data={disease_list_included_missing
                    .filter(d => d.status === 'Missing')
                    .map(({ id, name }) => ({ id, name }))}
                columns={['id', 'name']}
                pagination
                pageSize={10}
            />
        </Tab>
    </Tabs>
</Grid>

## Drug List Connection Overview

<Details title="Details">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This section visualizes how entities from the drug list connect to other categories in the knowledge graph.
The Sankey diagram shows the flow from incoming connection categories (left) through the drug list entities (center) 
to outgoing connection categories (right), providing insight into the types of knowledge graph relationships 
involving drug list entities.
</div>
</Details>

```sql drug_connections_sankey
-- Incoming connections: Subject Categories to Drug List
SELECT 
    concat('[IN] ', replace(subject_category,'biolink:','')) as source,
    'Drug' as target,
    sum(count) as count
FROM bq.drug_list_edges
WHERE direction = 'incoming'
GROUP BY subject_category
HAVING sum(count) > 20000

UNION ALL

-- Outgoing connections: Drug List to Object Categories
SELECT 
    'Drug' as source,
    concat('[OUT] ', replace(object_category,'biolink:','')) as target,
    sum(count) as count
FROM bq.drug_list_edges
WHERE direction = 'outgoing'
GROUP BY object_category
HAVING sum(count) > 20000
ORDER BY count DESC
```

<SankeyDiagram data={drug_connections_sankey} 
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  title='Drug List Connection Flow'
  subtitle='Flow from Incoming Categories through Drug List to Outgoing Categories (>20k connections)'
  chartAreaHeight={400}
/>

## Drug List Contents

<Details title="Details">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
This table shows all entities in the drug list with their connectivity information. The Edge Count column displays 
the total number of connections each drug has in the knowledge graph, providing insight into how well-connected 
each drug is within the broader network of biomedical knowledge. Click on any ID to access detailed information 
about that drug through the identifiers.org ID resolver.
</div>
</Details>

```sql drug_list_contents
SELECT id, name, edge_count, '<a href="http://identifiers.org/' || id || '" target="_blank">' || id || '</a>' as curie_link FROM bq.drug_list_nodes 
ORDER BY edge_count DESC
```

<DataTable 
    data={drug_list_contents} 
    search=true
    pagination=true
    title="Drug List Entities"
    link=url>
    
    <Column id="name" title="Name" />
    <Column id="curie_link" title="ID" contentType=html/>
    <Column id="edge_count" title="Edge Count" contentType="bar" />
</DataTable>
