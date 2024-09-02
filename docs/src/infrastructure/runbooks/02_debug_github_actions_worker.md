---
title: Fix Github Actions worker
---

We currently operate our own Github Actions worker. This is mostly for optimising speed of our pipeline. 
If something is wrong with the instance, you can SSH into it with the following command assuming you have the correct permissions:

```bash
gcloud compute ssh --zone "us-central1-c" "github-actions-runner" --tunnel-through-iap --project "mtrx-hub-dev-3of"
```

The instance is set up to prune docker every 7 days and restart once a week. 

Long term, we will likely migrate this setup to something kubernetes based but for the
time, this does the job. The machine is not terraformed or ansible'd in any codified way.