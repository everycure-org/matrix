with success_stats as (
  select
    id,
    split(id, ':')[offset(0)] as prefix -- Extract prefix from 'id'
  from `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized`
  where normalization_success = true
)

select
  prefix,
  count(*) as count,
  round(100 * count(*) / sum(count(*)) over(), 2) as percentage
from success_stats
group by prefix
order by count desc;