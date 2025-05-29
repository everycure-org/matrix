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

union all 

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