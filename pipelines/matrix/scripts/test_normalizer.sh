#!/bin/bash
# Convience script that gets normalized nodes from the nodenorm endpoint
set -xe
CURIES="$1"
IFS=',' read -ra CURIE_ARRAY <<< "$CURIES"
CURIE_JSON=$(printf '"%s",' "${CURIE_ARRAY[@]}" | sed 's/,$//')
# NOTE: This section was partially generated using AI assistance.
ENDPOINT="${2:-https://nodenorm.transltr.io/1.5/get_normalized_nodes}"

curl -X POST "$ENDPOINT" \
     -H "Content-Type: application/json" \
     -d '{
       "curies": ['$CURIE_JSON'],
       "conflate": true,
       "drug_chemical_conflate": true,
       "description": "true"
     }' | jq