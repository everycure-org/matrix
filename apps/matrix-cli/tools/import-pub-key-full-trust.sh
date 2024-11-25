#!/bin/bash

FILE_PATH=$1
echo "importing file $FILE_PATH"
# read the file
publicKey=$(cat $FILE_PATH)

fingerprint=$(echo "$publicKey" | gpg --show-keys | sed -n '2p' | xargs)
echo "The fingerprint is: $fingerprint"
echo "$publicKey" | gpg --import

echo "$fingerprint:6:" | gpg --import-ownertrust