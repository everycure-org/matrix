with normalization_stats as (
  select
    case
      when normalization_success then 'success'
      when original_id = id then 'miss'
      when original_id != id then 'miss_review'
    end as normalization_status
  from `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized`
)

select
  normalization_status,
  count(*) as count,
  round(100 * count(*) / sum(count(*)) over(), 2) as percentage
from normalization_stats
where normalization_status in ('success', 'miss')
group by normalization_status;
