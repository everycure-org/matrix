const duckdb = require('duckdb');
const path = require('path');

const dataFile = path.join(__dirname, 'temp_infores_catalog.jsonl');

const db = new duckdb.Database('sources/infores/infores.duckdb')
const conn = db.connect(); 

conn.all(`create or replace table catalog as select * from '${dataFile}'`, function(err, res) {
  if (err) {
    console.error(err);
    return;
  }
  console.log('Table created successfully');
  
  conn.close();
  db.close();
});
