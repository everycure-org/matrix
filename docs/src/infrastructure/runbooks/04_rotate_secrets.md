---
title: Rotating Git-Crypt Secrets
---
# Rotating Git-Crypt Secrets

This guide explains how to safely rotate Git-crypt secrets and manage access to encrypted repository content.

## Prerequisites

Before you begin, ensure you have:

1. Git-crypt installed and configured
2. GPG installed and configured
3. Access to the repository with git-crypt unlock privileges
4. The matrix-cli tool installed

## Rotating Out a User's Access

To remove a user's access to encrypted secrets, use the `matrix-cli secrets rotate-pk` command:

This command will:
1. Remove the user's public key from the repository
2. Verify all remaining users' keys are properly imported
3. Create a new private key
4. Re-encrypt secrets for remaining users

!!! warning
    - This is a destructive operation that requires explicit confirmation
    - Requires all remaining users' public keys to be present
    - Will automatically rollback changes if errors occur
    - the operator should triple check all secrets remain encrypted before pushing anything to the remote

## Best Practices

1. Consider creating a backup before rotating keys
2. Perform rotations on a feature branch
3. Only add the new secrets to the branch after the rotation is complete
4. Merge into main and notify people to pull before invalidating the old secrets to minimize disruption for people
4. Keep public keys synchronized using `sync-pks`
