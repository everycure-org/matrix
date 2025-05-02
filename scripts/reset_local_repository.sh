#!/bin/bash

# Ensure current branch is 'main'
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "main" ]; then
    echo "Error: You must be on 'main' to run this script. Current branch is '$current_branch'."
    exit 1
fi

# Fetch all remote changes
git fetch --all

# ask user to confirm
read -r -p "This next step wipes all local branches except main. Consider creating a copy of your repository if you want to ensure not loosing any work. [y/N] " response
if [[ "$response" =~ ^([yY])$ ]]; then
    # Delete all local branches except 'main'
    for branch in $(git branch | sed 's/\*//g' | awk '{$1=$1;print}' | grep -v "main"); do
        git branch -D "$branch"
done

# Hard reset 'main' to remote
git reset --hard origin/main

# Clean untracked files and directories
git clean -fd

# Optional: Prune old refs and optimize
git remote prune origin
git gc --prune=now