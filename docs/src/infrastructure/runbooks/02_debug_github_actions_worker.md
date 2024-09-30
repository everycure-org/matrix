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


## Out of disk space

We observed that the machine was out of disk space on Sept 30th. To avoid this in the future, a regular pruning has been set up

```
# For more information see the manual pages of crontab(5) and cron(8)
#
# m h  dom mon dow   command

0 5 * * 1 docker system prune -af | tee -a ~/.docker_prune.log
0 5 * * 1 docker volume prune -f | tee -a ~/.docker_prune.log
```

This lead to significant disk space cleanup

```
a3272bfa3f2c41a2ec3ddd1a2220d0c258734b2f2eb81ae35aef34620d13e581
68851265b55fb1281f15804aae58fffc9095ddc49ef431cc45f8acf358462b18
35d8780c4738d4bcd16f38d0ea19cfe27fe5f872ba5e2d1a2992c0ca8668e5a1
40d1c79c93931a3d77b3590837e28b128dc54e96df58f965c53aea4d9f83aa17
5ddc3bb76975bdfb898133c39e0ec4de714ea04a725c2dce951d4190924f74e0
66af754106d550692f7d861b3f6bfc46bb4a16cfa216b1a74b2c8de8f514a0b2

Total reclaimed space: 209GB
```

Additionally, a volume pruning step has been added to the CI pipeline to avoid running out of disk space.