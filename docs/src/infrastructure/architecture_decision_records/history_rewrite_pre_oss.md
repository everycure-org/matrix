---
title: Process for Open Sourcing MATRIX repository
---

## Context

A core goal of Every Cure and the MATRIX project is the open sourcing of the data as well as the
code we use to generate it. However, the repository was initially set up in a haste and contains
code that we do not have a license to open source as well as some information that we should not
release publicly such as encrypted secrets of our infrastructure. While these files are encrypted,
there are better ways to store them (e.g. in a separate, private repository).

## Decision

We have decided to perform the following steps to enable the OSS release:

1. Automatically review all comments of the repository using an LLM prompted for risk flagging. This
   is done using a utility function written to extract all comments and review comments
2. Extract all encrypted secrets to a separate repository `matrix-secrets` which will be added as a
   submodule to the main matrix repository (and thus cannot be cloned by OSS contributors)
3. Extract all libraries to a separate repo `matrix-legacy-proprietary-packages` which is added as a
   submodule until the library replacement is complete. This will block OSS contributors from using
   the fabricated data pipeline (`test`).
4. Rewrite all history to remove the above resources from the repository
5. Rotate all secrets

### Automated Comment Review

A command was added to the `matrix` CLI in `apps/matrix-cli` which allows for automated scanning of
all comments. This library sends each comment on all issues and PRs to a Gemini model for
inspection. Critical comments are flagged to the user and can be manually reviewed and/or deleted
until we see no further risks from our comments.

All comments deemed problematic have been stored in [this sheet](https://docs.google.com/spreadsheets/d/1EWi0HTh5gSAawXcJFqnCo4GKIk4HaM-i/edit?usp=sharing&ouid=103803540614117230127&rtpof=true&sd=true) for the team to review and potentially mitigate.
Generally, we only found some banter & jokes, but it is still a good idea to review and potentially
tidy up our language before we open source. 

### Secret Extraction

Most secrets have already been moved to GCP Secrets Manager. However something has to _bootstrap_
the secrets and these bootstrapped values are stored in an encrypted file which is shared with the
respective environment administrators (dev/prod). We continue to deploy `git-crypt` for this but
move these encrypted files into a separate repository to minimize risk exposure. The
separate repository is added as a submodule which is only accessible to Every Cure
org members. 

#### Rotate all secrets

All secrets stored in `infra/secrets` should be rotated and the old values invalidated after a grace
period. This ensures even if our history rewrite is faulty, all secrets have been rotated out and
are no longer valid.

### Library Extraction

The key library to extract is `data_fabricator` which is extracted into a separate repository. We
can simply move this library to another repo and then replace the `pipelines/matrix/packages` folder
with a submodule reference to this repository.
[PR 1325](https://github.com/everycure-org/matrix/pull/1325) contains this change.

### History Rewrite

1. Close or merge all open PRs. This will require careful coordination with the team to ensure we
   have a moment where all open work has been completed.
1. Create a full mirror of the repository in a separate, private repository called `matrix-backup`.
   This is a safety precaution and ideally will not be used further.
1. Perform a local history rewrite with `git-filter-repo` to remove key folders and files from any
   commit in the repository.
1. Push the changes to Github, overwriting all existing branches with the new history chain
1. Ask all current maintainers to reset their local repository to the new history chain using
   `scripts/reset_local_repository.sh`


## Consequences

- PRs and issues referencing commit IDs will be broken. This is an unavoidable side effect of
  the history rewrite. We still maintain all git history which is what we want to achieve.
- Administrators need to clone with submodules included to unlock the secrets.
- Documentation needs to be added to enable OSS contributors to still install the pipeline even
  without access to the fabricator

## Benefits

After this work, the entire mono repository can be open sourced. We can also remove
authentication from our documentation page at this stage and freely share links with
people to get outside perspectives and contributions.

## References

- [Github documentation on history rewrites for private information](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
