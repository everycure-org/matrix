const duckdb = require('duckdb');

const db = new duckdb.Database('sources/valid_edge_types/valid_edge_types.duckdb')

const conn = db.connect(); 

conn.all("create or replace table valid_edge_types as select * from 'https://github.com/everycure-org/matrix-schema/releases/latest/download/valid_biolink_edge_types.tsv'", function(err, res) {
  if (err) {
    console.error(err);
    return;
  }
});

conn.close();
db.close();