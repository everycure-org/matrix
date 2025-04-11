/**
 * Sets the run_id in the environment for the Evidence dashboard.
 * This script is used during the Evidence Dashboard deployment process to
 * set the correct run-specific parameters.
 */

const fs = require('fs');
const path = require('path');

const run_id = process.argv[2];
if (!run_id) {
    console.error('Please provide a run_id as an argument');
    process.exit(1);
}

// Update evidence.config.yaml
const configPath = path.join(__dirname, '../evidence.config.yaml');
let configContent = fs.readFileSync(configPath, 'utf8');
configContent = configContent.replace(
    /(basePath:\s*\/runs\/)[^/\n]*/,
    `$1${run_id}`
);
fs.writeFileSync(configPath, configContent);

// Create .env file with run_id using EVIDENCE_VAR__ prefix
const envPath = path.join(__dirname, '../.env');
fs.writeFileSync(envPath, `EVIDENCE_VAR__run_id=${run_id}\n`);

console.log(`Set run_id to ${run_id}`);

// NOTE: This file was partially generated using AI assistance. 