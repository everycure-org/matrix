---
title: Create a Release
---

---
title: Create a Release
---

## Overview

This runbook outlines the steps to create a release in our GitHub repository.

## Steps to Create a Release

1. **Create Release through GitHub**
    - We create releases through GitHub.
    - The template for the release notes is kept in `.github/release.yml`.
2. **Tagging and Versioning**
    - We publish tags following [Semantic Versioning](https://semver.org/) in our GitHub repository and release based on these tags. If any breaking changes exist, we need to bump a major version. For releases `<v1.0` it should be a minor bump instead
4. **Release Content**
    - The releases contain the following key sections:
      - New code and features implemented.
      - Experiments completed (based on "Experiment Report" merged PRs).
      - Data & Matrix prediction versions published to BigQuery.
5. **Prune Merged PRs**
    - Before executing the release, ensure all merged PRs are correctly labeled and have intuitive titles. Update them if necessary. 
    - Also ensure they are correctly labelled
    - if PRs were created by Person A but really owned by Person B, make sure you call out Person B in the PR, not the one that created it
6. **Generate Release Notes Draft**
    - Generate a release notes draft.
    - Review the list and check the names of the PRs to ensure they read nicely. Consider reshuffling them so they tell a good story instead of just being a list of things.
7. **List Contributors**
    - Manually check who has contributed and list the contributors of the month to encourage contributions through PRs (code, docs, experiment reports, etc.). See the cli command below for how to best do this

## Commands

To list contributors, use the following command:

```bash
git log v0.1..HEAD --pretty=format:"%h %ae%n%b" | \
    awk '/^[0-9a-f]+ / {hash=$1; author=$2; print hash, author} /^Co-authored-by:/ {if (match($0, /<[^>]+>/)) print hash, substr($0, RSTART+1, RLENGTH-2)}' | \
    awk '{print $2}' | \
    sort -u
```


## Best Practices

- Ensure all PRs are labeled and titled correctly before generating the release notes.
- Review and update the release notes draft to ensure clarity and completeness.
- Acknowledge all contributors to foster a collaborative environment.

## Additional Resources

- [Semantic Versioning](https://semver.org/)
- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
