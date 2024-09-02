---
title: Create a Release
---

- we create releases through Github 
- in `.github/release.yml` we keep the template for the release notes 
- we publish tags following semver versioning in our github repository and release based on these tags
- the releases contain the following key sections
  - new code and features implemented
  - experiments completed (based on "Experiment Report" merged PRs)
  - [TODO] Data & Matrix prediction versions published to BigQuery
- before executing on the release, we prune the PRs that have been merged by ensuring they are all correctly labeled and have intuitive titles
- we also manually check who has contributed and list the contributors of the month to encourage people to materialize their contributions through some form of PR (code, docs, experiment report etc)
    ```
    git log v0.1..HEAD --pretty=format:"%h %ae%n%b" |                                                                                                                                                      3075ms
      awk '/^[0-9a-f]+ / {hash=$1; author=$2; print hash, author}
      /^Co-authored-by:/ {if (match($0, /<[^>]+>/)) print hash, substr($0, RSTART+1, RLENGTH-2)}' |
      awk '{print $2}' |
      sort | uniq
    ```