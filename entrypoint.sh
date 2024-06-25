#!/bin/sh

cypher-shell -a ${NEO4J_ADRESS} -u ${NEO4J_USER} -p ${NEO4J_PASS} -f /tmp/statup-script.cql

echo "Neo4J boostrapped..."

trap : TERM INT
sleep infinity &
wait