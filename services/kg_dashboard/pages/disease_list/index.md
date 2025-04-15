```sql disease_list
select *, '/disease_list/' || id as link from bq.disease_list
```

<DataTable data={disease_list}
  title="Disease List"
  search={true}
  pagination={true}
  rows={20}
  link=link
>
  <Column id="name" title="Disease" />
  <Column id="edge_count" contentType="bar" title="Edge Count" />
</DataTable>