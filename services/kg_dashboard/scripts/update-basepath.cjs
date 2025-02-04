// NOTE: This script was partially generated using AI assistance.
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