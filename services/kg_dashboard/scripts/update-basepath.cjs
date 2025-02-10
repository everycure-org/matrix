/**
 * Updates the basePath in evidence.config.yaml based on a version argument.
 * This script is used during the Evidence Dashboard deployment process to
 * set the correct version-specific base URL path for the dashboard.
 * Not doing this breaks the paths in the static website files.
 * This can be removed once evidence.dev supports 
 * basePath as a runtime variable
 * https://github.com/evidence-dev/evidence/discussions/3068
 */

const fs = require('fs');
const path = require('path');

const version = process.argv[2];
const configPath = path.join(__dirname, '../evidence.config.yaml');

let content = fs.readFileSync(configPath, 'utf8');
content = content.replace(
    /(basePath:\s*\/versions\/)[^/\n]*/,
    `$1${version}`
);
fs.writeFileSync(configPath, content);