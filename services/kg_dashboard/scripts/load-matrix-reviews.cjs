const duckdb = require('duckdb');
const path = require('path');

const dataFile = path.join(__dirname, 'temp_matrix_reviews.csv');

const db = new duckdb.Database('sources/matrix_reviews/matrix_reviews.duckdb')

const conn = db.connect();

conn.all(`create or replace table relevancy_scores as select * from '${dataFile}'`, function(err, res) {
  if (err) {
    console.error(err);
    return;
  }
  console.log('Table created successfully');
});

conn.close();
db.close();
