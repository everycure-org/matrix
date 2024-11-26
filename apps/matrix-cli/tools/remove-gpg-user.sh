#!/bin/bash
#
# Script to remove GPG key from git-crypt
# original located at: https://gist.github.com/glogiotatidis/e0ab45ed5575a9d7973390dace0552b0
#
# It will re-initialize git-crypt for the repository and re-add all keys except
# the one requested for removal.
#
# Note: You still need to change all your secrets to fully protect yourself.
# Removing a user will prevent them from reading future changes but they will
# still have a copy of the data up to the point of their removal.
#
# Use:
#  ./remove-gpg-user.sh [FULL_GPG_FINGERPRINT]
#
# E.g.:
#  ./remove-gpg-user.sh 3BC18383F838C0B815B961480F8CAF5467D
#
# The script will create multiple commits to your repo. Feel free to squash them
# all down to one.
#
# Based on https://github.com/AGWA/git-crypt/issues/47#issuecomment-212734882
#
#
set -xe

if [ -z "$1" ]
then
    echo " Use:"
    echo "  ./remove-gpg-user.sh [FULL_GPG_FINGERPRINT] [KEY_NAME]"
    echo ""
    echo " E.g.:"
    echo "  ./remove-gpg-user.sh 3BC18383F838C0B815B961480F8CAF5467D my-key-name"
    exit;
fi
# if no KEY_NAME is provided, we assume 'default'
if [ -z "$2" ]
then
    KEY_NAME="default"
fi

TMPDIR=`mktemp -d`
CURRENT_DIR=`git rev-parse --show-toplevel`
BASENAME=$(basename `pwd`)

# Unlock the directory, we need to copy encrypted versions of the files
git crypt unlock

# Work on copy.
cp -rp `pwd` $TMPDIR


pushd $TMPDIR/$BASENAME

# Remove encrypted files and git-crypt
git crypt status | grep -v "not encrypted" > encrypted-files
awk '{print $2}' encrypted-files | xargs rm
git commit -a -m "Remove encrypted files"
rm -rf .git-crypt
git commit -a -m "Remove git-crypt"
rm -rf .git/git-crypt

# Delete the specific user's GPG key file from the original directory
rm -f "$CURRENT_DIR/.git-crypt/keys/$KEY_NAME/0/$1.gpg"

# Re-initialize git crypt
git crypt init

# Add existing users, except the person we removed
for keyfilename in `ls $CURRENT_DIR/.git-crypt/keys/$KEY_NAME/0/*gpg`; do
    basename=`basename $keyfilename`
    key=${basename%.*}
    if [[ $key == $1 ]]; then
        continue;
    fi
    git crypt add-gpg-user $key -k $KEY_NAME
done

cd $CURRENT_DIR
for i in `awk '{print $2}' ${TMPDIR}/${BASENAME}/encrypted-files`; do
    rsync -R $i $TMPDIR/$BASENAME;
done
cd $TMPDIR/$BASENAME
for i in `awk '{print $2}' encrypted-files`; do
    git add $i
done
git commit -a -m "New encrypted files"
popd

git crypt lock -a
git pull $TMPDIR/$BASENAME

rm -rf $TMPDIR