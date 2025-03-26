# {params.disease}

Title: {params.disease} Dashboard

```sql disease
select * from bq.disease_list
where id = '${params.disease}'
```

name: <strong><Value data={disease} column="name" /></strong>

<Value data={disease} column="definition" />
