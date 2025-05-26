select 
  'disease_list' as source
  , original_id
  , id
  , name
  , category
from 
  `mtrx-hub-dev-3of.release_${bq_release_version}.disease_list_nodes_normalized`
where 
  original_id != id

union all 

select 
  'drug_list' as source
  , original_id
  , id
  , name
  , category
from 
  `mtrx-hub-dev-3of.release_${bq_release_version}.drug_list_nodes_normalized` 
where 
  original_id != id