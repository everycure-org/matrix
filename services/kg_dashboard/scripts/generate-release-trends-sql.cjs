#!/usr/bin/env node

// Script to generate the release_trends.sql file dynamically
// based on available BigQuery datasets
const { BigQuery } = require('@google-cloud/bigquery');

async function generateReleaseTrendsSQL() {
  const projectId = process.env.PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT;
  const currentReleaseVersion = process.env.RELEASE_VERSION || 'v0.9.9';
  
  if (!projectId) {
    console.error('Error: PROJECT_ID or GOOGLE_CLOUD_PROJECT environment variable not set');
    process.exit(1);
  }

  console.log(`Filtering releases up to current version: ${currentReleaseVersion}`);

  const bigquery = new BigQuery({ projectId });

  try {
    // Query to find all release datasets (only standard vX_Y_Z pattern, not experimental ones)
    const query = `
      SELECT 
        schema_name as dataset_id,
        REGEXP_EXTRACT(schema_name, r'^release_(v[0-9]+_[0-9]+_[0-9]+)$') as bq_version,
        REPLACE(REGEXP_EXTRACT(schema_name, r'^release_(v[0-9]+_[0-9]+_[0-9]+)$'), '_', '.') as semantic_version
      FROM \`${projectId}.INFORMATION_SCHEMA.SCHEMATA\`
      WHERE REGEXP_CONTAINS(schema_name, r'^release_v[0-9]+_[0-9]+_[0-9]+$')
      ORDER BY schema_name
    `;

    const [rows] = await bigquery.query(query);
    
    if (rows.length === 0) {
      console.error('No release datasets found');
      process.exit(1);
    }

    // Parse version numbers for proper numeric sorting and filtering
    const parseVersion = (version) => {
      const parts = version.replace('v', '').split('.').map(num => parseInt(num, 10));
      return parts;
    };
    
    const compareVersions = (a, b) => {
      const aVersion = parseVersion(a);
      const bVersion = parseVersion(b);
      
      // Compare major, minor, patch in order
      for (let i = 0; i < Math.max(aVersion.length, bVersion.length); i++) {
        const aPart = aVersion[i] || 0;
        const bPart = bVersion[i] || 0;
        
        if (aPart !== bPart) {
          return aPart - bPart;
        }
      }
      
      return 0;
    };

    // Filter out releases that are newer than the current release version
    const filteredReleases = rows.filter(row => {
      if (!row.bq_version || !row.semantic_version) {
        return false;
      }
      
      // Only include releases that are <= current release version
      return compareVersions(row.semantic_version, currentReleaseVersion) <= 0;
    });

    // Now verify each release has nodes/edges tables by trying to count from them
    console.log(`Checking ${filteredReleases.length} releases for nodes/edges tables...`);
    const validReleases = [];
    
    for (const release of filteredReleases) {
      let nodesTable = null;
      let edgesTable = null;
      let diseaseTable = null;
      let drugTable = null;
      
      // Try different table name conventions for nodes
      const nodeTables = ['nodes_unified', 'unified_nodes'];
      for (const tableName of nodeTables) {
        try {
          const testQuery = `SELECT 1 FROM \`${projectId}.${release.dataset_id}.${tableName}\` LIMIT 1`;
          await bigquery.query(testQuery);
          nodesTable = tableName;
          break;
        } catch (error) {
          // Table doesn't exist or isn't accessible, try next one
        }
      }
      
      // Try different table name conventions for edges  
      const edgeTables = ['edges_unified', 'unified_edges'];
      for (const tableName of edgeTables) {
        try {
          const testQuery = `SELECT 1 FROM \`${projectId}.${release.dataset_id}.${tableName}\` LIMIT 1`;
          await bigquery.query(testQuery);
          edgesTable = tableName;
          break;
        } catch (error) {
          // Table doesn't exist or isn't accessible, try next one
        }
      }
      
      // Check for disease_list_nodes_normalized table
      try {
        const testQuery = `SELECT 1 FROM \`${projectId}.${release.dataset_id}.disease_list_nodes_normalized\` LIMIT 1`;
        await bigquery.query(testQuery);
        diseaseTable = 'disease_list_nodes_normalized';
      } catch (error) {
        // Table doesn't exist or isn't accessible
      }
      
      // Check for drug_list_nodes_normalized table
      try {
        const testQuery = `SELECT 1 FROM \`${projectId}.${release.dataset_id}.drug_list_nodes_normalized\` LIMIT 1`;
        await bigquery.query(testQuery);
        drugTable = 'drug_list_nodes_normalized';
      } catch (error) {
        // Table doesn't exist or isn't accessible
      }
      
      if (nodesTable && edgesTable) {
        release.nodes_table = nodesTable;
        release.edges_table = edgesTable;
        release.disease_table = diseaseTable;
        release.drug_table = drugTable;
        validReleases.push(release);
        console.log(`  ✓ ${release.semantic_version} - found ${nodesTable}, ${edgesTable}${diseaseTable ? ', ' + diseaseTable : ''}${drugTable ? ', ' + drugTable : ''}`);
      } else {
        console.log(`  Skipping ${release.semantic_version} - missing tables (nodes: ${nodesTable}, edges: ${edgesTable})`);
      }
    }
    
    // Sort releases by semantic version (numeric sort, not alphabetic)
    validReleases.sort((a, b) => compareVersions(a.semantic_version, b.semantic_version));
    
    console.log(`Found ${validReleases.length} releases with nodes/edges tables:`, validReleases.map(r => r.semantic_version).join(', '));

    // Generate the SQL with direct counting from nodes/edges tables
    let sql;
    if (validReleases.length > 0) {
      const unionClauses = validReleases.map((release, index) => {
        const drugMedianQuery = release.drug_table ? `(
        with edge_count_per_subject as (
          select subject, count(*) as n_edges_per_subject
          from \`\${project_id}.${release.dataset_id}.${release.edges_table}\`
          group by 1
        ),
        edge_count_per_object as (
          select object, count(*) as n_edges_per_object
          from \`\${project_id}.${release.dataset_id}.${release.edges_table}\`
          where object != subject
          group by 1
        ),
        edge_count as (
          select subject as id, n_edges_per_subject as n_edges from edge_count_per_subject
          union all
          select object as id, n_edges_per_object as n_edges from edge_count_per_object
        ),
        nodes_degrees as (
          select id, sum(n_edges) as degree
          from edge_count
          group by 1
        )
        select PERCENTILE_CONT(n.degree, 0.5) OVER() as median
        from nodes_degrees n
        inner join \`\${project_id}.${release.dataset_id}.${release.drug_table}\` dr on n.id = dr.id
        limit 1
      )` : 'cast(null as float64)';
      
        const diseaseMedianQuery = release.disease_table ? `(
        with edge_count_per_subject as (
          select subject, count(*) as n_edges_per_subject
          from \`\${project_id}.${release.dataset_id}.${release.edges_table}\`
          group by 1
        ),
        edge_count_per_object as (
          select object, count(*) as n_edges_per_object
          from \`\${project_id}.${release.dataset_id}.${release.edges_table}\`
          where object != subject
          group by 1
        ),
        edge_count as (
          select subject as id, n_edges_per_subject as n_edges from edge_count_per_subject
          union all
          select object as id, n_edges_per_object as n_edges from edge_count_per_object
        ),
        nodes_degrees as (
          select id, sum(n_edges) as degree
          from edge_count
          group by 1
        )
        select PERCENTILE_CONT(n.degree, 0.5) OVER() as median
        from nodes_degrees n
        inner join \`\${project_id}.${release.dataset_id}.${release.disease_table}\` di on n.id = di.id
        limit 1
      )` : 'cast(null as float64)';

        return `    select 
      '${release.bq_version}' as bq_version,
      '${release.semantic_version}' as semantic_version,
      ${index + 1} as release_order,
      (select count(*) from \`\${project_id}.${release.dataset_id}.${release.nodes_table}\`) as n_nodes,
      (select count(*) from \`\${project_id}.${release.dataset_id}.${release.edges_table}\`) as n_edges,
      (select count(distinct primary_knowledge_source) from \`\${project_id}.${release.dataset_id}.${release.edges_table}\`) as n_distinct_knowledge_sources,
      ${release.disease_table ? `(select count(*) from \`\${project_id}.${release.dataset_id}.${release.disease_table}\`)` : 'cast(null as int64)'} as n_nodes_from_disease_list,
      ${release.drug_table ? `(select count(*) from \`\${project_id}.${release.dataset_id}.${release.drug_table}\`)` : 'cast(null as int64)'} as n_nodes_from_drug_list,
      ${drugMedianQuery} as median_drug_node_degree,
      ${diseaseMedianQuery} as median_disease_node_degree`;
      }).join('\n    union all\n');

      sql = `-- Get key metrics across all available releases to show trends over time
-- This file is auto-generated by scripts/generate-release-trends-sql.js
-- Last generated: ${new Date().toISOString()}
-- Using direct counts from nodes/edges tables

with all_release_metrics as (
${unionClauses}
)`;
    } else {
      // Fallback: create a query that returns no results but has the right schema
      console.log('No accessible release tables found, creating fallback query...');
      sql = `-- Get key metrics across all available releases to show trends over time
-- This file is auto-generated by scripts/generate-release-trends-sql.js
-- Last generated: ${new Date().toISOString()}
-- NOTE: No accessible release datasets found, returning empty results

with all_release_metrics as (
  select 
    cast(null as string) as bq_version,
    cast(null as string) as semantic_version,
    cast(null as int64) as release_order,
    cast(null as int64) as n_nodes,
    cast(null as int64) as n_edges,
    cast(null as int64) as n_distinct_knowledge_sources,
    cast(null as int64) as n_nodes_from_disease_list,
    cast(null as int64) as n_nodes_from_drug_list,
    cast(null as float64) as median_drug_node_degree,
    cast(null as float64) as median_disease_node_degree
  where false  -- This ensures no rows are returned
)`;
    }

    sql += `

select 
  semantic_version,
  bq_version,
  release_order,
  -- Mark current release
  case when bq_version = '\${bq_release_version}' then true else false end as is_current_release,
  
  n_nodes,
  n_edges,
  n_distinct_knowledge_sources,
  n_edges / n_nodes as edges_per_node,
  n_nodes_from_disease_list,
  n_nodes_from_drug_list,
  median_drug_node_degree,
  median_disease_node_degree,
  
  -- Calculate changes from previous release
  n_nodes - lag(n_nodes) over (order by release_order) as nodes_change,
  n_edges - lag(n_edges) over (order by release_order) as edges_change,
  
  -- Calculate percentage changes from previous release  
  round(100.0 * (n_nodes - lag(n_nodes) over (order by release_order)) / lag(n_nodes) over (order by release_order), 2) as nodes_pct_change,
  round(100.0 * (n_edges - lag(n_edges) over (order by release_order)) / lag(n_edges) over (order by release_order), 2) as edges_pct_change

from 
  all_release_metrics
order by 
  release_order`;

    // Write the SQL file
    const fs = require('fs');
    const path = require('path');
    
    const sqlPath = path.join(__dirname, '..', 'sources', 'bq', 'release_trends.sql');
    fs.writeFileSync(sqlPath, sql);
    
    console.log(`Generated release_trends.sql with ${validReleases.length} releases`);

  } catch (error) {
    console.error('Error generating SQL:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  generateReleaseTrendsSQL();
}

module.exports = { generateReleaseTrendsSQL };