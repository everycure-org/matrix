const duckdb = require('duckdb');

const db = new duckdb.Database('sources/infores/infores.duckdb')

const conn = db.connect(); 

conn.all("create or replace table catalog as select * from 'https://github.com/biolink/information-resource-registry/releases/latest/download/infores_catalog.jsonl'", function(err, res) {
  if (err) {
    console.error(err);
    return;
  }
});

conn.close();
db.close();
