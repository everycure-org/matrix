const duckdb = require('duckdb');
const path = require('path');

const dataFile = path.join(__dirname, 'temp_valid_edge_types.tsv');

const db = new duckdb.Database('sources/valid_edge_types/valid_edge_types.duckdb')

const conn = db.connect(); 

conn.all(`create or replace table valid_edge_types as select * from '${dataFile}'`, function(err, res) {
  if (err) {
    console.error(err);
    return;
  }
  console.log('Table created successfully');
});

conn.close();
db.close();