# Restoring a backup

Our cluster has a schedule enabled for [automatic backups](https://cloud.google.com/kubernetes-engine/docs/add-on/backup-for-gke/how-to/backup-schedule), this guide describes how to restore a backup.

## Manually creating a backup

If the backup needs to be created on-demand, you should go to the [GKE backups page](https://console.cloud.google.com/kubernetes/backups/backupPlans) and Click the dropdown icon on "More Actions" and select "Start a backup". You can leave all values default, only give your backup a name.

> TODO: Shall we limit backups to Grafana/MLFlow/Ledger namespaces?

## Defining a restore plan


A [restore plan](https://cloud.google.com/kubernetes-engine/docs/add-on/backup-for-gke/how-to/restore-plan) defines how the backup is restored.

```bash
gcloud beta container backup-restore restore-plans create neo4j-plan \
    --project=mtrx-hub-prod-sms \
    --location=us-central1 \
    --backup-plan=projects/mtrx-hub-prod-sms/locations/us-central1/backupPlans/rpo-daily-window \
    --cluster=projects/mtrx-hub-prod-sms/locations/us-central1/clusters/compute-cluster \
    --namespaced-resource-restore-mode=merge-replace-on-conflict \
    --selected-namespaces=neo4j \
    --cluster-resource-conflict-policy=use-existing-version \
    --cluster-resource-scope-selected-group-kinds=cluster-resource-scope-no-group-kinds \
    --volume-data-restore-policy=restore-volume-data-from-backup
```

## Restore backup

```bash
gcloud beta container backup-restore restores create restore-neo4j \
    --project=mtrx-hub-prod-sms \
    --location=us-central1 \
    --restore-plan=neo4j-plan \
    --backup=projects/mtrx-hub-prod-sms/locations/us-central1/backupPlans/rpo-daily-window/backups/neo4j-test-backup \
    --wait-for-completion
```

> 🆘 For the backup to take effect, we had to run a `neo4j stop` in the Neo4j shell, otherwise the database remained in the `starting` state.