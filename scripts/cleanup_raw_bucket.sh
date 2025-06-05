#! /bin/bash

# This script is used as part of CI to clean up the raw bucket from old files. We 
# do not want to give people delete files permissions to avoid manipulation of RAW data. 
# However several people have a desire to clean up old files. This script & the CI job
# enables this while keeping a 4 eye process to avoid uncontrolled deletion. 

PATHS=$(cat scripts/cleanup_files.txt | grep -v -E '^(#|$)')
for path in $PATHS; do
    echo "Deleting $path"
    gsutil -m rm -rf $path
done
