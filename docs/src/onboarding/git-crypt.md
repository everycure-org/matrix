# Git-Crypt

Our repository uses [git-crypt](https://www.agwa.name/projects/git-crypt/) to encrypt secrets in our code repository. This way, everything one needs to access the data is contained in the repository
without actually exposing the data itself because a decryption key is required.

<div style="position: relative; width: 100%; height: 0; padding-bottom: 56.25%;"><iframe src="https://us06web.zoom.us/clips/embed/fAnKtX_JDGeJAZgjqlvODUuQuSZ18FRO08Ia9yEdxbS_zMm07xgBMUtY4JrStCKb3gOuZWg-jCyAlgG-2SikYOQA.Qa3KXZe2qmtNyJSH" frameborder="0" allowfullscreen="allowfullscreen" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; "></iframe></div>

## Required installations

=== "MacOS"

    ```bash
    brew install git-crypt
    brew install gpg
    ```

=== "Windows (WSL)"

    ```bash
    sudo apt-get update
    sudo apt-get install git-crypt
    ```

## Joining as a new user

As a new user execute the following command to generate your [gpg-key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

### Generate the key
To generate a new key, run the following command, we recommend sticking to the defaults
everywhere. If you already use gpg for signing your git commits, feel free to share your
existing public key (use the gpg command with all default options). 

Also, do not select any of the options they ask you. Continue pressing enter until you arrive at the prompt: "Key does not expire at all". Select "y" for that. 

```bash
gpg --full-generate-key
```

!!! tip
    GPG will ask you for a password to protect your key. We recommend skipping this unless you have a strong opinion about having a passphrase. If you do add one, please remember it as we will not be able to provide it to you if you forget it. (On some versions of GPG, you can't skip this step. Make sure to write down your passphrase so you don't forget it. You'll need it to unlock the git-crypt). 

??? info "Example command output"

    ```
    $ gpg --full-generate-key
    gpg (GnuPG) 2.4.5; Copyright (C) 2024 g10 Code GmbH
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.

    Please select what kind of key you want:
       (1) RSA and RSA
       (2) DSA and Elgamal
       (3) DSA (sign only)
       (4) RSA (sign only)
       (9) ECC (sign and encrypt) *default*
      (10) ECC (sign only)
      (14) Existing key from card
    Your selection?
    Please select which elliptic curve you want:
       (1) Curve 25519 *default*
       (4) NIST P-384
       (6) Brainpool P-256
    Your selection?
    Please specify how long the key should be valid.
             0 = key does not expire
          <n>  = key expires in n days
          <n>w = key expires in n weeks
          <n>m = key expires in n months
          <n>y = key expires in n years
    Key is valid for? (0)
    Key does not expire at all
    Is this correct? (y/N) y

    GnuPG needs to construct a user ID to identify your key.

    Real name: Test Man
    Email address: test@man.org
    Comment:
    You selected this USER-ID:
        "Test Man <test@man.org>"

    Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit? O
    We need to generate a lot of random bytes. It is a good idea to perform
    some other action (type on the keyboard, move the mouse, utilize the
    disks) during the prime generation; this gives the random number
    generator a better chance to gain enough entropy.
    gpg: WARNING: server 'gpg-agent' is older than us (2.2.41 < 2.4.5)
    gpg: Note: Outdated servers may lack important security fixes.
    gpg: Note: Use the command "gpgconf --kill all" to restart them.
    We need to generate a lot of random bytes. It is a good idea to perform
    some other action (type on the keyboard, move the mouse, utilize the
    disks) during the prime generation; this gives the random number
    generator a better chance to gain enough entropy.
    gpg: revocation certificate stored as '/Users/pascalwhoop/.gnupg/openpgp-revocs.d/637334785B9F74A364C14FC5A1C5DB60575ED571.rev'
    public and secret key created and signed.

    pub   ed25519 2024-08-05 [SC]
          637334785B9F74A364C14FC5A1C5DB60575ED571
    uid                      Test Man <test@man.org>
    sub   cv25519 2024-08-05 [E]
    ```

### Copy the key

Next run the following command

```bash
# alternatively, you can pass in the ID of the key which looks like this 637334785B9F74A364C14FC5A1C5DB60575ED571
gpg --armor --export used@email.com
```

The command should return something like 

```asc
-----BEGIN PGP PUBLIC KEY BLOCK-----

mDMEZqysGBYJKwYBBAHaRw8BAQdAHc+eXOKkeeycxWcL5iVj68LjpRZEsTMIHr+g
W4YWF7S0GlRlc3RtYW4gPHRlc3RAZXhhbXBsZS5jb20+iJMEExYKADsWIQSrkwB+
g/2d1UeZ+HN2DBhFT8RGvAUCZqysGAIbIwULCQgHAgIiAgYVCgkICwIEFgIDAQIe
BwIXgAAKCRB2DBhFT8RGvHBnAQDquAfciyO+b2U1+1yLyppqIYzP110RuydemCE5
KtNQIAEAvRQ6jqDyNl4iIaa9mv2qxEJF182ajO5Br6sgoAJv7wI=
=lfMd
-----END PGP PUBLIC KEY BLOCK-----
```

1. Copy the entire key 
    - **including** `--BEGIN PGP PUBLIC KEY BLOCK--` 
    - **and** `--END PGP PUBLIC KEY BLOCK--` 
2. Create a PR with the key in the `.git-crypt/keys/public_keys` folder, name the file like `your-github-username.asc`
3. Tag one of the team leads to review the PR[^1]

??? question "What happens behind the scenes?"

    An existing team member will to add the key to the repository. What they do is run the following command
    
    ```bash
    sh import_gpg_key.sh
    ```
    
    This will encrypt a shared secret key with your public key and store the encrypted key in
    the repository. This will make you uniquely able to decrypt the file without sharing the
    secret key itself with anyone else. The files are stored in `.git-crypt/keys/` for those curious. 


[As you wait, learn about kedro! :material-skip-next:](./kedro.md){ .md-button .md-button--primary }

### Unlocking the key once your key was added

Once the public key was added to the repo (i.e. the PR was merged and the team lead added the encrypted shared secret key), you can unlock the key by running

```bash
git-crypt unlock
```

Re-enter the passphrase you used to create the public key to unlock. 

You can lock the key by running 
```
git-crypt lock -a
``` 

## General setup

We have 2 keys in the repository:

- `HUB` key: used to decrypt the SA to read the data on GCS from the hub project.
- `default` key: used to decrypt additional infra secrets used to set up the infrastructure. Not shared with everyone

Most people should not need to decrypt the `default` key and can stick to the `HUB` key.
The CD system uses the `default` key to decrypt the secrets needed to deploy the
infrastructure.


---

## Additional reading

- [Git Crypt Introduction](https://www.agwa.name/projects/git-crypt/)
- [Git Crypt with multiple keys](https://stackoverflow.com/questions/77187053/how-to-use-git-crypt-with-multiple-keys)


[^1]: The team lead will take your key and add an encrypted version of our shared secret key to the repository which only you can decrypt
