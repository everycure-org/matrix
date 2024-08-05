# Git-Crypt

Our repository uses [git-crypt](https://www.agwa.name/projects/git-crypt/) to materialize secrets in our code repository.

## Required installations

```bash
brew install git-crypt
brew install gpg
```

## Joining as a new user

As a new user execute the following command to generate your [gpg-key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

```bash
export EMAIL="your@email.com"
export NAME="Your Name"
gpg --quick-generate-key "${NAME} <${EMAIL}>" ed25519 sign,auth never
gpg --armor --export $EMAIL
```

The command should return something like 

```
-----BEGIN PGP PUBLIC KEY BLOCK-----

mDMEZqysGBYJKwYBBAHaRw8BAQdAHc+eXOKkeeycxWcL5iVj68LjpRZEsTMIHr+g
W4YWF7S0GlRlc3RtYW4gPHRlc3RAZXhhbXBsZS5jb20+iJMEExYKADsWIQSrkwB+
g/2d1UeZ+HN2DBhFT8RGvAUCZqysGAIbIwULCQgHAgIiAgYVCgkICwIEFgIDAQIe
BwIXgAAKCRB2DBhFT8RGvHBnAQDquAfciyO+b2U1+1yLyppqIYzP110RuydemCE5
KtNQIAEAvRQ6jqDyNl4iIaa9mv2qxEJF182ajO5Br6sgoAJv7wI=
=lfMd
-----END PGP PUBLIC KEY BLOCK-----
```

Copy the entire text and share it with the team leads (check the CODEOWNERS file for guidance)

## Unlocking the key once your key was added

Once the public key was added to the repo, you can unlock the key by running

```bash
git-crypt unlock
```



## Actions by the existing admins

Once the new user has shared their public key, the admins need to add the key to the repository. A convenience script exists at `scripts/import_gpg_key.sh`

```bash
sh import_gpg_key.sh
```

We have 2 keys in the repository:

- `HUB` key: used to decrypt the SA to read the data on GCS from the hub project.
- `default` key: used to decrypt additional infra secrets used to set up the infrastructure. Not shared with everyone

## Additional reading

- https://stackoverflow.com/questions/77187053/how-to-use-git-crypt-with-multiple-keys
- https://github.com/AGWA/git-crypt