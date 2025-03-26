---
title: Release Article Template Generation
---

## Overview

This runbook explains the cases in which the release article template is generated from the GitHub Actions workflow of kedro pipeline submission. The template only outlines the PR titles and url links of each category, classified by labels. The golden rule is that the article is expected only when there is a **major** or **minor** bump.

## Examples
Given that the latest minor release is v0.2.5:

1. **Major bump**  
    - New release: v1.0.0 (major bump). Article template generation is expected.
    
2. **Minor bump**  
    - New release: v0.3.0 (minor bump). Article template generation is expected.
    
3. **Patch bump**  
    - New release: v0.2.8 (patch bump). Article template generation is skipped.
    
4. **Patch downgrade**  
    - New release: v0.2.3 (patch downgrade). Article template generation is skipped.  
    - Heads up: If the release version order is v0.2.5 → v0.3.2 → v0.3.0, **v0.3.2** is considered a minor bump, as it is triggered first, and **v0.3.0** is considered a patch downgrade, as it follows v0.3.2.
    
5. **Major or minor downgrade**  
    - New release: v0.1.8 (minor downgrade). This is forbidden and prevented from `kedro experiment run`. The same applies to major downgrades.
The reason to forbid it is that is a bit weird: yes, the Python programming language allows it, but for good reasons. Whereas we might add bugfixes to older releases, all new feature development should continue from the latest official release. processing the commits and PRs that have lead to this intermediate release, while not taking into account features that were added "in the future" (relative to this intermediate release) is harder if a major or minor release would be created that isn't the latest, then the release notes and article of the already existing release would become unlogical, since they are written relative to the last official (major or minor) release. An intermediate (major or minor) release would need to mention contributions that were already included in the latest release, and would need to remove it from that one too.

## Commands

To generate the template manually, first activate the virtual environment by running 

```bash
source ./matrix/apps/matrix-cli/.venv/bin/activate
```

Then use the following command:

```bash
matrix releases template --output-file <OUTPUT_LOCATION> \
    --since <GIT_SHA_START> \
    --until <GIT_SHA_END> 
```

The `--since` and `--until` flags define the range of commits to include in the template. When using the `--headless` flag, the starting Git SHA will default to the latest minor release tag, the ending Git SHA will default to the current branch. If the `--since` flag is omitted, you will be prompted to select a release from a list.

## Best Practices

- Ensure all PRs are correctly labeled and titled before generating the article release template.
- Acknowledge all contributors to foster a collaborative environment.