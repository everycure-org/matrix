#!/bin/bash

# Script to list GPG key IDs and associated emails from git-crypt keys

# Directory containing the git-crypt keys
KEYS_DIR=".git-crypt/keys/HUB/0"

# Loop through each .gpg file in the keys directory
for key_file in "$KEYS_DIR"/*.gpg; do
    # Extract the key ID from the filename (remove path and .gpg extension)
    key_id=$(basename "$key_file" .gpg)
    
    # Print the key ID
    echo -n "$key_id: "
    
    # Use gpg to get key information
    # --list-keys: list keys
    # --with-colons: output in an easy-to-parse format
    # Then use awk to extract the email (field 10) from the uid line
    # 'exit' after the first match to only get the primary email
    gpg --list-keys --with-colons "$key_id" | awk -F: '/^uid:/ {print $10; exit}'
done

# Note: This script assumes that gpg is installed and the keys are in your keyring
# If a key is not found, gpg will print an error message for that key
