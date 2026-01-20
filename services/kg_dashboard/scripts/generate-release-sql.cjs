#!/usr/bin/env node

// Unified script to generate both release_trends.sql and key_nodes_release_trends.sql
// Discovers BigQuery datasets once and generates both SQL files
// Also provides KG version extraction from globals.yml for dashboard display
const { BigQuery } = require('@google-cloud/bigquery');
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

// ============================================================================
// Shared Release Discovery Logic
// ============================================================================

/**
 * Parse semantic version string into numeric array for comparison
 */
function parseVersion(version) {
  const parts = version.replace('v', '').split('.').map(num => parseInt(num, 10));
  return parts;
}

/**
 * Compare two semantic versions
 * Returns: -1 if a < b, 0 if a == b, 1 if a > b
 */
function compareVersions(a, b) {
  const aVersion = parseVersion(a);
  const bVersion = parseVersion(b);

  for (let i = 0; i < Math.max(aVersion.length, bVersion.length); i++) {
    const aPart = aVersion[i] || 0;
    const bPart = bVersion[i] || 0;

    if (aPart !== bPart) {
      return aPart - bPart;
    }
  }

  return 0;
}

/**
 * Check if a version is a major release (patch version is 0)
 * Used by filterToKeyReleases to reduce query costs by limiting version scans
 * Note: This is for release filtering, not KG source filtering
 */
function isMajorRelease(semanticVersion) {
  const parts = parseVersion(semanticVersion);
  return parts.length >= 3 && parts[2] === 0;
}

/**
 * Filter releases to only major releases + benchmark + current
 * This reduces query costs by limiting the number of versions scanned
 */
function filterToKeyReleases(validReleases, benchmarkVersion, currentReleaseVersion) {
  const keyReleases = validReleases.filter(release => {
    // Keep major releases (x.y.0)
    if (isMajorRelease(release.semantic_version)) {
      return true;
    }
    // Keep benchmark version
    if (release.semantic_version === benchmarkVersion) {
      return true;
    }
    // Keep current release version
    if (release.semantic_version === currentReleaseVersion) {
      return true;
    }
    return false;
  });

  // Remove duplicates (in case benchmark or current is also a major release)
  const seen = new Set();
  return keyReleases.filter(release => {
    if (seen.has(release.semantic_version)) {
      return false;
    }
    seen.add(release.semantic_version);
    return true;
  });
}

/**
 * Discover all valid release datasets in BigQuery
 * Filters to releases <= currentReleaseVersion
 * Validates that each release has required tables
 */
async function discoverReleases(bigquery, projectId, currentReleaseVersion) {
  console.log(`\n=== Discovering Release Datasets ===`);
  console.log(`Filtering releases up to current version: ${currentReleaseVersion}`);

  // Query to find all release datasets
  const query = `
    SELECT
      schema_name as dataset_id,
      REGEXP_EXTRACT(schema_name, r'^release_(v[0-9]+_[0-9]+_[0-9]+)$') as bq_version,
      REPLACE(REGEXP_EXTRACT(schema_name, r'^release_(v[0-9]+_[0-9]+_[0-9]+)$'), '_', '.') as semantic_version
    FROM \`${projectId}.INFORMATION_SCHEMA.SCHEMATA\`
    WHERE REGEXP_CONTAINS(schema_name, r'^release_v[0-9]+_[0-9]+_[0-9]+$')
    ORDER BY schema_name
  `;

  const [releases] = await bigquery.query(query);

  if (releases.length === 0) {
    console.error('No release datasets found');
    process.exit(1);
  }

  // Filter releases <= current version
  const filteredReleases = releases.filter(row => {
    if (!row.bq_version || !row.semantic_version) {
      return false;
    }
    return compareVersions(row.semantic_version, currentReleaseVersion) <= 0;
  });

  // Validate each release has required tables
  console.log(`Checking ${filteredReleases.length} releases for required tables...`);
  const validReleases = [];

  for (const release of filteredReleases) {
    let nodesTable = null;
    let edgesTable = null;
    let diseaseTable = null;
    let drugTable = null;

    // Try different table name conventions for nodes
    const nodeTables = ['nodes_unified', 'nodes'];
    for (const tableName of nodeTables) {
      try {
        const testQuery = `SELECT 1 FROM \`${projectId}.${release.dataset_id}.${tableName}\` LIMIT 1`;
        await bigquery.query(testQuery);
        nodesTable = tableName;
        break;
      } catch (error) {
        // Table doesn't exist, try next one
      }
    }

    // Try different table name conventions for edges
    const edgeTables = ['edges_unified', 'edges'];
    for (const tableName of edgeTables) {
      try {
        const testQuery = `SELECT 1 FROM \`${projectId}.${release.dataset_id}.${tableName}\` LIMIT 1`;
        await bigquery.query(testQuery);
        edgesTable = tableName;
        break;
      } catch (error) {
        // Table doesn't exist, try next one
      }
    }

    // Check for disease_list_nodes_normalized table
    try {
      const testQuery = `SELECT 1 FROM \`${projectId}.${release.dataset_id}.disease_list_nodes_normalized\` LIMIT 1`;
      await bigquery.query(testQuery);
      diseaseTable = 'disease_list_nodes_normalized';
    } catch (error) {
      // Table doesn't exist
    }

    // Check for drug_list_nodes_normalized table
    try {
      const testQuery = `SELECT 1 FROM \`${projectId}.${release.dataset_id}.drug_list_nodes_normalized\` LIMIT 1`;
      await bigquery.query(testQuery);
      drugTable = 'drug_list_nodes_normalized';
    } catch (error) {
      // Table doesn't exist
    }

    // Check which primary knowledge source field exists
    let pksField = null;
    if (edgesTable) {
      try {
        const testQuery = `SELECT primary_knowledge_sources FROM \`${projectId}.${release.dataset_id}.${edgesTable}\` LIMIT 1`;
        await bigquery.query(testQuery);
        pksField = 'primary_knowledge_sources';
      } catch (error) {
        try {
          const testQuery = `SELECT primary_knowledge_source FROM \`${projectId}.${release.dataset_id}.${edgesTable}\` LIMIT 1`;
          await bigquery.query(testQuery);
          pksField = 'primary_knowledge_source';
        } catch (e) {
          pksField = null;
        }
      }
    }

    if (nodesTable && edgesTable) {
      release.nodes_table = nodesTable;
      release.edges_table = edgesTable;
      release.disease_table = diseaseTable;
      release.drug_table = drugTable;
      release.pks_field = pksField;
      validReleases.push(release);
      console.log(`  ✓ ${release.semantic_version} - found ${nodesTable}, ${edgesTable}${diseaseTable ? ', ' + diseaseTable : ''}${drugTable ? ', ' + drugTable : ''} [pks: ${pksField || 'none'}]`);
    } else {
      console.log(`  Skipping ${release.semantic_version} - missing tables (nodes: ${nodesTable}, edges: ${edgesTable})`);
    }
  }

  // Sort releases by semantic version
  validReleases.sort((a, b) => compareVersions(a.semantic_version, b.semantic_version));

  console.log(`Found ${validReleases.length} valid releases:`, validReleases.map(r => r.semantic_version).join(', '));

  return validReleases;
}

// ============================================================================
// KG Pipeline Metrics Discovery and Generation
// ============================================================================

/**
 * Exclusion list for non-KG data sources
 */
const KG_EXCLUSIONS = [
  'disease_list',
  'drug_list',
  'ec_ground_truth',
  'ec_clinical_trials',
  'kgml_xdtd_ground_truth',
  'off_label'
];

/**
 * Display name mapping for KGs
 */
const KG_DISPLAY_NAMES = {
  'rtx_kg2': 'RTX-KG2',
  'robokop': 'ROBOKOP',
  'primekg': 'PrimeKG'
};

// ============================================================================
// KG Version Extraction from globals.yml
// ============================================================================

/**
 * Upstream knowledge graph sources (explicit inclusion list)
 * These are the sources that appear in the Knowledge Graphs dashboard section
 */
const KG_SOURCES = [
  'rtx_kg2',
  'robokop',
  'primekg',
  'spoke',
  'embiology'
];

/**
 * Path to globals.yml relative to this script
 */
const GLOBALS_YML_PATH = path.join(__dirname, '..', '..', '..', 'pipelines', 'matrix', 'conf', 'base', 'globals.yml');

/**
 * Extract KG versions from globals.yml
 * Returns an object mapping KG source names to their versions
 * Only includes sources in the KG_SOURCES list (actual upstream knowledge graphs)
 */
function extractKGVersions() {
  try {
    const globalsContent = fs.readFileSync(GLOBALS_YML_PATH, 'utf8');
    const globals = yaml.load(globalsContent);

    const dataSources = globals.data_sources || {};
    const kgVersions = {};

    for (const [sourceName, sourceConfig] of Object.entries(dataSources)) {
      // Only include known KG sources
      if (!KG_SOURCES.includes(sourceName)) {
        continue;
      }

      // Extract version if present
      if (sourceConfig && sourceConfig.version !== undefined) {
        // Convert version to string (handles numeric versions like 2.1)
        kgVersions[sourceName] = String(sourceConfig.version);
      }
    }

    return kgVersions;
  } catch (error) {
    console.error(`Error reading globals.yml: ${error.message}`);
    return {};
  }
}

/**
 * Print KG versions as JSON to stdout (for Makefile capture)
 * Used by: $(shell node -e "require('./scripts/generate-release-sql.cjs').printKGVersions()")
 */
function printKGVersions() {
  const versions = extractKGVersions();
  // Output JSON without newlines so it works in Makefile
  process.stdout.write(JSON.stringify(versions));
}

/**
 * Generate kg_versions.sql with version info for each KG
 * This creates a simple SQL file that returns version data without querying BigQuery
 */
function generateKGVersionsSQL() {
  console.log(`\n=== Generating kg_versions.sql ===`);

  const versions = extractKGVersions();
  const kgNames = Object.keys(versions);

  if (kgNames.length === 0) {
    console.log('No KG versions found in globals.yml');
    return `-- KG Versions from globals.yml
-- Auto-generated by generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}
-- NOTE: No KG versions found

SELECT
  CAST(NULL AS STRING) as knowledge_graph,
  CAST(NULL AS STRING) as display_name,
  CAST(NULL AS STRING) as version
WHERE FALSE
`;
  }

  const unionClauses = kgNames.map(kg =>
    `  SELECT '${kg}' as knowledge_graph, '${getKGDisplayName(kg)}' as display_name, '${versions[kg]}' as version`
  );

  const sql = `-- KG Versions from globals.yml
-- Auto-generated by generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}

${unionClauses.join('\n  UNION ALL\n')}
ORDER BY knowledge_graph
`;

  console.log(`Generated kg_versions.sql with ${kgNames.length} KG versions: ${kgNames.join(', ')}`);
  return sql;
}

/**
 * Get display name for a KG
 */
function getKGDisplayName(kgSource) {
  if (KG_DISPLAY_NAMES[kgSource]) {
    return KG_DISPLAY_NAMES[kgSource];
  }
  // Convert snake_case to Title Case
  return kgSource.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

/**
 * Discover all KG sources and their available tables in the current release dataset
 * Returns: Array of { kg_source, display_name, tables: { nodes_ingested, nodes_transformed, nodes_normalized, edges_ingested, edges_transformed, edges_normalized } }
 */
async function discoverKGSources(bigquery, projectId, datasetId) {
  console.log(`\n=== Discovering KG Sources in ${datasetId} ===`);

  // Query to find all KG-related tables
  const query = `
    SELECT table_name
    FROM \`${projectId}.${datasetId}.INFORMATION_SCHEMA.TABLES\`
    WHERE (
      table_name LIKE '%_nodes_ingested%'
      OR table_name LIKE '%_nodes_transformed'
      OR table_name LIKE '%_nodes_normalized'
      OR table_name LIKE '%_edges_ingested%'
      OR table_name LIKE '%_edges_transformed'
      OR table_name LIKE '%_edges_normalized'
    )
    ORDER BY table_name
  `;

  const [tables] = await bigquery.query(query);

  if (tables.length === 0) {
    console.log('No KG tables found');
    return [];
  }

  // Group tables by KG source
  const kgSources = {};

  for (const row of tables) {
    const tableName = row.table_name;

    // Extract KG source from table name
    // Pattern: {kg_source}_{entity_type}_{stage}[_{version}]
    // Examples: rtx_kg2_nodes_ingested_v2_10_0, robokop_edges_transformed, primekg_nodes_normalized

    let kgSource = null;
    let entityType = null;
    let stage = null;

    // Try to match ingested tables (with version suffix)
    const ingestedMatch = tableName.match(/^(.+?)_(nodes|edges)_ingested_(.+)$/);
    if (ingestedMatch) {
      kgSource = ingestedMatch[1];
      entityType = ingestedMatch[2];
      stage = 'ingested';
    }

    // Try to match transformed/normalized tables (no version suffix)
    const otherMatch = tableName.match(/^(.+?)_(nodes|edges)_(transformed|normalized)$/);
    if (otherMatch) {
      kgSource = otherMatch[1];
      entityType = otherMatch[2];
      stage = otherMatch[3];
    }

    if (!kgSource || !entityType || !stage) {
      console.log(`  Skipping unrecognized table: ${tableName}`);
      continue;
    }

    // Skip excluded sources
    if (KG_EXCLUSIONS.includes(kgSource)) {
      continue;
    }

    // Initialize KG source entry if needed
    if (!kgSources[kgSource]) {
      kgSources[kgSource] = {
        kg_source: kgSource,
        display_name: getKGDisplayName(kgSource),
        tables: {}
      };
    }

    // Store table name
    const tableKey = `${entityType}_${stage}`;
    kgSources[kgSource].tables[tableKey] = tableName;
  }

  const result = Object.values(kgSources);
  console.log(`Found ${result.length} KG sources:`);
  for (const kg of result) {
    const stages = Object.keys(kg.tables).join(', ');
    console.log(`  ✓ ${kg.display_name} (${kg.kg_source}): ${stages}`);
  }

  return result;
}

/**
 * Generate kg_pipeline_metrics.sql with counts for each KG's pipeline stages
 */
function generateKGPipelineMetricsSQL(kgSources, projectId, datasetId) {
  console.log(`\n=== Generating kg_pipeline_metrics.sql ===`);

  if (kgSources.length === 0) {
    return `-- KG Pipeline Metrics
-- This file is auto-generated by scripts/generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}
-- NOTE: No KG sources found, returning empty results

SELECT
  CAST(NULL AS STRING) as knowledge_graph,
  CAST(NULL AS STRING) as display_name,
  CAST(NULL AS STRING) as entity_type,
  CAST(NULL AS STRING) as stage,
  CAST(NULL AS INT64) as sort_order,
  CAST(NULL AS INT64) as count
WHERE FALSE
`;
  }

  const unionClauses = [];

  for (const kg of kgSources) {
    // For each entity type (nodes, edges)
    for (const entityType of ['nodes', 'edges']) {
      // For each stage in order
      const stages = [
        { name: 'Ingested', key: `${entityType}_ingested`, sort_order: 1 },
        { name: 'Transformed', key: `${entityType}_transformed`, sort_order: 2 },
        { name: 'Normalized', key: `${entityType}_normalized`, sort_order: 3 }
      ];

      for (const stage of stages) {
        const tableName = kg.tables[stage.key];

        if (tableName) {
          // Table exists, count rows
          unionClauses.push(`  SELECT
    '${kg.kg_source}' as knowledge_graph,
    '${kg.display_name}' as display_name,
    '${entityType}' as entity_type,
    '${stage.name}' as stage,
    ${stage.sort_order} as sort_order,
    (SELECT COUNT(*) FROM \`\${project_id}.${datasetId}.${tableName}\`) as count`);
        }
        // If table doesn't exist, we simply don't include a row for that stage
        // This handles KGs that don't have ingested tables
      }
    }
  }

  const sql = `-- KG Pipeline Metrics - counts per KG/stage/entity type
-- This file is auto-generated by scripts/generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}

WITH metrics AS (
${unionClauses.join('\n\n  UNION ALL\n\n')}
)

SELECT
  knowledge_graph,
  display_name,
  entity_type,
  stage,
  sort_order,
  count
FROM metrics
ORDER BY knowledge_graph, entity_type, sort_order
`;

  console.log(`Generated kg_pipeline_metrics.sql for ${kgSources.length} KG sources`);
  return sql;
}

// ============================================================================
// Release Trends SQL Generation
// ============================================================================

function generateReleaseTrendsSQL(validReleases, projectId) {
  console.log(`\n=== Generating release_trends.sql ===`);

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
-- This file is auto-generated by scripts/generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}
-- Using direct counts from nodes/edges tables

with all_release_metrics as (
${unionClauses}
)`;
  } else {
    sql = `-- Get key metrics across all available releases to show trends over time
-- This file is auto-generated by scripts/generate-release-sql.cjs
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
  where false
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

  console.log(`Generated release_trends.sql with ${validReleases.length} releases`);
  return sql;
}

// ============================================================================
// Key Node Release Trends SQL Generation
// ============================================================================

function generateKeyNodeTrendsSQL(validReleases, projectId, keyNodeIds) {
  console.log(`\n=== Generating key_nodes_release_trends.sql ===`);
  console.log(`Tracking ${keyNodeIds.length} key nodes: ${keyNodeIds.join(', ')}`);

  let sql;
  if (validReleases.length > 0) {
    // Build CTEs for descendants for each release
    const descendantCTEs = validReleases.map((release) => {
      // Build primary knowledge source filter based on available field
      let pksFilter = '';
      if (release.pks_field === 'primary_knowledge_sources') {
        pksFilter = `  AND EXISTS(
    SELECT 1 FROM UNNEST(edges.primary_knowledge_sources.list) AS pks
    WHERE pks.element IN ('infores:mondo', 'infores:chebi')
  )`;
      } else if (release.pks_field === 'primary_knowledge_source') {
        pksFilter = `  AND edges.primary_knowledge_source IN ('infores:mondo', 'infores:chebi')`;
      }

      return `descendants_${release.bq_version} AS (
  SELECT
    key_node_ids.id as key_node_id,
    key_node_ids.id as descendant_id,
    0 as depth
  FROM (
    SELECT id FROM UNNEST(SPLIT('\${key_disease_ids}', ',')) AS id
    UNION ALL
    SELECT id FROM UNNEST(SPLIT('\${key_drug_ids}', ',')) AS id
  ) key_node_ids

  UNION ALL

  SELECT
    descendants_${release.bq_version}.key_node_id,
    edges.subject as descendant_id,
    descendants_${release.bq_version}.depth + 1 as depth
  FROM descendants_${release.bq_version}
  JOIN \`\${project_id}.${release.dataset_id}.${release.edges_table}\` edges
    ON edges.object = descendants_${release.bq_version}.descendant_id
    AND edges.predicate = 'biolink:subclass_of'
  WHERE descendants_${release.bq_version}.depth < 20
${pksFilter}
)`;
    }).join(',\n\n');

    // Build SELECT queries for each release
    const unionClauses = validReleases.map((release, index) => {
      return `SELECT
  descendants_${release.bq_version}.key_node_id,
  '${release.bq_version}' as bq_version,
  '${release.semantic_version}' as semantic_version,
  ${index + 1} as release_order,
  edges.predicate,
  subject_nodes.category as subject_category,
  object_nodes.category as object_category,
  edges.primary_knowledge_source,
  COUNT(*) as edge_count,
  COUNT(DISTINCT edges.subject) as unique_subjects,
  COUNT(DISTINCT edges.object) as unique_objects
FROM descendants_${release.bq_version}
JOIN \`\${project_id}.${release.dataset_id}.${release.edges_table}\` edges
  ON edges.subject = descendants_${release.bq_version}.descendant_id
     OR edges.object = descendants_${release.bq_version}.descendant_id
JOIN \`\${project_id}.${release.dataset_id}.${release.nodes_table}\` subject_nodes
  ON edges.subject = subject_nodes.id
JOIN \`\${project_id}.${release.dataset_id}.${release.nodes_table}\` object_nodes
  ON edges.object = object_nodes.id
GROUP BY
  descendants_${release.bq_version}.key_node_id,
  edges.predicate,
  subject_nodes.category,
  object_nodes.category,
  edges.primary_knowledge_source`;
    }).join('\n\nUNION ALL\n\n');

    sql = `-- Key node edge type trends across releases
-- This file is auto-generated by scripts/generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}
-- Tracks edge type changes for key node neighborhoods across releases

WITH RECURSIVE
${descendantCTEs}

${unionClauses}

ORDER BY key_node_id, release_order, edge_count DESC
`;
  } else {
    sql = `-- Key node edge type trends across releases
-- This file is auto-generated by scripts/generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}
-- NOTE: No accessible release datasets found, returning empty results

SELECT
  CAST(NULL AS STRING) as key_node_id,
  CAST(NULL AS STRING) as bq_version,
  CAST(NULL AS STRING) as semantic_version,
  CAST(NULL AS INT64) as release_order,
  CAST(NULL AS STRING) as predicate,
  CAST(NULL AS STRING) as subject_category,
  CAST(NULL AS STRING) as object_category,
  CAST(NULL AS STRING) as primary_knowledge_source,
  CAST(NULL AS INT64) as edge_count,
  CAST(NULL AS INT64) as unique_subjects,
  CAST(NULL AS INT64) as unique_objects
WHERE FALSE
`;
  }

  console.log(`Generated key_nodes_release_trends.sql with ${validReleases.length} releases`);
  return sql;
}

// ============================================================================
// Key Node Release Aggregate SQL Generation
// ============================================================================

function generateKeyNodeAggregateSQL(validReleases, projectId, keyNodeIds) {
  console.log(`\n=== Generating key_nodes_release_aggregate.sql ===`);
  console.log(`Tracking ${keyNodeIds.length} key nodes: ${keyNodeIds.join(', ')}`);

  let sql;
  if (validReleases.length > 0) {
    // Build CTEs for descendants for each release (same as trends)
    const descendantCTEs = validReleases.map((release) => {
      // Build primary knowledge source filter based on available field
      let pksFilter = '';
      if (release.pks_field === 'primary_knowledge_sources') {
        pksFilter = `  AND EXISTS(
    SELECT 1 FROM UNNEST(edges.primary_knowledge_sources.list) AS pks
    WHERE pks.element IN ('infores:mondo', 'infores:chebi')
  )`;
      } else if (release.pks_field === 'primary_knowledge_source') {
        pksFilter = `  AND edges.primary_knowledge_source IN ('infores:mondo', 'infores:chebi')`;
      }

      return `descendants_${release.bq_version} AS (
  SELECT
    key_node_ids.id as key_node_id,
    key_node_ids.id as descendant_id,
    0 as depth
  FROM (
    SELECT id FROM UNNEST(SPLIT('\${key_disease_ids}', ',')) AS id
    UNION ALL
    SELECT id FROM UNNEST(SPLIT('\${key_drug_ids}', ',')) AS id
  ) key_node_ids

  UNION ALL

  SELECT
    descendants_${release.bq_version}.key_node_id,
    edges.subject as descendant_id,
    descendants_${release.bq_version}.depth + 1 as depth
  FROM descendants_${release.bq_version}
  JOIN \`\${project_id}.${release.dataset_id}.${release.edges_table}\` edges
    ON edges.object = descendants_${release.bq_version}.descendant_id
    AND edges.predicate = 'biolink:subclass_of'
  WHERE descendants_${release.bq_version}.depth < 20
${pksFilter}
)`;
    }).join(',\n\n');

    // Build aggregate SELECT queries for each release
    const unionClauses = validReleases.map((release, index) => {
      return `SELECT
  descendants_${release.bq_version}.key_node_id,
  '${release.bq_version}' as bq_version,
  '${release.semantic_version}' as semantic_version,
  ${index + 1} as release_order,
  COUNT(DISTINCT CASE
    WHEN descendants_${release.bq_version}.descendant_id = descendants_${release.bq_version}.key_node_id
    THEN edges.subject || '|' || edges.predicate || '|' || edges.object
  END) as direct_total_edges,
  COUNT(DISTINCT edges.subject || '|' || edges.predicate || '|' || edges.object) as with_descendants_total_edges
FROM descendants_${release.bq_version}
JOIN \`\${project_id}.${release.dataset_id}.${release.edges_table}\` edges
  ON edges.subject = descendants_${release.bq_version}.descendant_id
     OR edges.object = descendants_${release.bq_version}.descendant_id
GROUP BY
  descendants_${release.bq_version}.key_node_id`;
    }).join('\n\nUNION ALL\n\n');

    sql = `-- Key node aggregate edge counts across releases
-- This file is auto-generated by scripts/generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}
-- Tracks total edge counts (direct and with descendants) for key nodes across releases

WITH RECURSIVE
${descendantCTEs}

${unionClauses}

ORDER BY key_node_id, release_order
`;
  } else {
    sql = `-- Key node aggregate edge counts across releases
-- This file is auto-generated by scripts/generate-release-sql.cjs
-- Last generated: ${new Date().toISOString()}
-- NOTE: No accessible release datasets found, returning empty results

SELECT
  CAST(NULL AS STRING) as key_node_id,
  CAST(NULL AS STRING) as bq_version,
  CAST(NULL AS STRING) as semantic_version,
  CAST(NULL AS INT64) as release_order,
  CAST(NULL AS INT64) as direct_total_edges,
  CAST(NULL AS INT64) as with_descendants_total_edges
WHERE FALSE
`;
  }

  console.log(`Generated key_nodes_release_aggregate.sql with ${validReleases.length} releases`);
  return sql;
}

// ============================================================================
// Main Execution
// ============================================================================

async function main() {
  const projectId = process.env.EVIDENCE_VAR__project_id;
  const currentReleaseVersion = process.env.EVIDENCE_VAR__release_version;
  const bqReleaseVersion = process.env.EVIDENCE_VAR__bq_release_version;
  const benchmarkVersion = process.env.EVIDENCE_VAR__benchmark_version;
  const keyDiseaseIds = process.env.EVIDENCE_VAR__key_disease_ids || '';
  const keyDrugIds = process.env.EVIDENCE_VAR__key_drug_ids || '';

  // Validate environment
  if (!projectId) {
    console.error('Error: EVIDENCE_VAR__project_id environment variable not set');
    process.exit(1);
  }

  if (!currentReleaseVersion) {
    console.error('Error: EVIDENCE_VAR__release_version environment variable not set');
    process.exit(1);
  }

  if (!bqReleaseVersion) {
    console.error('Error: EVIDENCE_VAR__bq_release_version environment variable not set');
    process.exit(1);
  }

  const allKeyNodeIds = [
    ...keyDiseaseIds.split(',').filter(id => id.trim()),
    ...keyDrugIds.split(',').filter(id => id.trim())
  ];

  if (allKeyNodeIds.length === 0) {
    console.error('Error: No key node IDs configured');
    process.exit(1);
  }

  const bigquery = new BigQuery({ projectId });
  const currentDatasetId = `release_${bqReleaseVersion}`;

  try {
    // Step 1: Discover all valid releases
    const allValidReleases = await discoverReleases(bigquery, projectId, currentReleaseVersion);

    // Step 2: Filter to key releases (major + benchmark + current) to reduce query costs
    const validReleases = filterToKeyReleases(allValidReleases, benchmarkVersion, currentReleaseVersion);
    console.log(`\n=== Filtered to Key Releases ===`);
    console.log(`Reduced from ${allValidReleases.length} to ${validReleases.length} releases`);
    console.log(`Key releases: ${validReleases.map(r => r.semantic_version).join(', ')}`);

    // Step 3: Generate release_trends.sql
    const releaseTrendsSQL = generateReleaseTrendsSQL(validReleases, projectId);
    const releaseTrendsPath = path.join(__dirname, '..', 'sources', 'bq', 'release_trends.sql');
    fs.writeFileSync(releaseTrendsPath, releaseTrendsSQL);

    // Step 4: Generate key_nodes_release_trends.sql
    const keyNodeTrendsSQL = generateKeyNodeTrendsSQL(validReleases, projectId, allKeyNodeIds);
    const keyNodeTrendsPath = path.join(__dirname, '..', 'sources', 'bq', 'key_nodes_release_trends.sql');
    fs.writeFileSync(keyNodeTrendsPath, keyNodeTrendsSQL);

    // Step 5: Generate key_nodes_release_aggregate.sql
    const keyNodeAggregateSQL = generateKeyNodeAggregateSQL(validReleases, projectId, allKeyNodeIds);
    const keyNodeAggregatePath = path.join(__dirname, '..', 'sources', 'bq', 'key_nodes_release_aggregate.sql');
    fs.writeFileSync(keyNodeAggregatePath, keyNodeAggregateSQL);

    // Step 6: Discover KG sources and generate kg_pipeline_metrics.sql
    const kgSources = await discoverKGSources(bigquery, projectId, currentDatasetId);
    const kgPipelineMetricsSQL = generateKGPipelineMetricsSQL(kgSources, projectId, currentDatasetId);
    const kgPipelineMetricsPath = path.join(__dirname, '..', 'sources', 'bq', 'kg_pipeline_metrics.sql');
    fs.writeFileSync(kgPipelineMetricsPath, kgPipelineMetricsSQL);

    // Step 7: Generate kg_versions.sql from globals.yml
    const kgVersionsSQL = generateKGVersionsSQL();
    const kgVersionsPath = path.join(__dirname, '..', 'sources', 'bq', 'kg_versions.sql');
    fs.writeFileSync(kgVersionsPath, kgVersionsSQL);

    console.log(`\n=== Success ===`);
    console.log(`Generated 5 SQL files from ${validReleases.length} releases and ${kgSources.length} KG sources`);

  } catch (error) {
    console.error('Error generating SQL:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = {
  discoverReleases,
  discoverKGSources,
  generateReleaseTrendsSQL,
  generateKeyNodeTrendsSQL,
  generateKeyNodeAggregateSQL,
  generateKGPipelineMetricsSQL,
  generateKGVersionsSQL,
  extractKGVersions,
  printKGVersions
};
