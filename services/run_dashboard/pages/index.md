---
title: Matrix Pair Prediction Run Dashboard
---

<script>
  const run_id = import.meta.env.EVIDENCE_VAR__run_id;
</script>


# Distribution
<Image 
    url="https://storage.cloud.google.com/mtrx-us-central1-hub-dev-storage/dev_work/image.png"
    description="Matrix histogram"
/>Distribution of treat scores, not treat scores, and unknown scores


```sql drug_scores
  select 
      *
    FROM bq.drug_scores_per_n
``` 
# Drug Scores

<DataTable data={drug_scores} search=true link=link title='Drug Scores per n (Frequent Flyers)'>

	<Column id='drug_id' />
	<Column id=count_in_1000 title="In top 1000" contentType=bar/>
	<Column id=count_in_10000 title="In top 10000" contentType=bar barColor=#aecfaf/>
	<Column id=count_in_100000 title="In top 100000" contentType=bar barColor=#53768a/>
	<Column id='drug_name'/>
	<Column id='mean_treat_score'/>
	<Column id='root_mean_square_treat_score'/>
	<Column id='max_treat_score'/>

</DataTable>

```sql disease_scores
  select 
      *
    FROM bq.disease_scores_per_n
```

# Disease Scores

<DataTable data={disease_scores} search=true link=link title='Disease Scores per n (Frequent Flyers)'>

	<Column id='disease_id' />
	<Column id=count_in_1000 title="In top 1000" contentType=bar/>
	<Column id=count_in_10000 title="In top 10000" contentType=bar barColor=#aecfaf/>
	<Column id=count_in_100000 title="In top 100000" contentType=bar barColor=#53768a/>
	<Column id='disease_name'/>
	<Column id='mean_treat_score'/>
	<Column id='root_mean_square_treat_score'/>
	<Column id='max_treat_score'/>

</DataTable>
