with normalization_stats as (
  select
    normalization_success,
    case
      when original_id = id and normalization_success then 'success'
      when original_id = id and not normalization_success then 'miss'
      when original_id != id and normalization_success then 'success'
      when original_id != id and not normalization_success then 'miss_review'
    end as normalization_status
  from `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized`
)

select
  normalization_status,
  count(*) as count,
  round(100 * count(*) / sum(count(*)) over(), 2) as percentage
from normalization_stats
where normalization_status in ('success', 'miss')
group by normalization_status;

with success_stats as (
  select
    id,
    split(id, ':')[offset(0)] as prefix -- Extract prefix from 'id'
  from `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized`
  where normalization_success = true
)

select
  prefix,
  count(*) as count,
  round(100 * count(*) / sum(count(*)) over(), 2) as percentage
from success_stats
group by prefix
order by count desc;