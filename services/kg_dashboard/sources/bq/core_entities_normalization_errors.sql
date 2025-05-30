(
  select 
    'disease_list' as source
    , original_id
    , id
    , name
    , category
    , normalization_success
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized`
  where 
    original_id != id
    or not normalization_success
  limit ${max_core_entities_normalization_errors}
)

union all 

(
  select 
    'drug_list' as source
    , original_id
    , id
    , name
    , category
    , normalization_success
  from 
    `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` 
  where 
    original_id != id
    or not normalization_success
  limit ${max_core_entities_normalization_errors}
)