#!/bin/bash

# Prompt the admin to tell us which git-crypt key to share with
echo "Please share the name of the key to share (see git-crypt -k)"
read keyName

# Prompt the user to enter the public key
echo "Please enter the public GPG key (press Ctrl-D when done):"
publicKey=$(</dev/stdin)

fingerprint=$(echo "$publicKey" | gpg --show-keys | sed -n '2p' | xargs)
echo "The fingerprint is: $fingerprint"
echo "$publicKey" | gpg --import

echo "$fingerprint:6:" | gpg --import-ownertrust
git-crypt add-gpg-user -k $keyName $fingerprint