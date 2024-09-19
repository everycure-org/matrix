#!/bin/bash
set -xe

# origin/run/23-aug-setup-1
# origin/run/23-aug-setup-2
# origin/run/23-aug-setup-3
# origin/run/23-aug-setup-4
# origin/run/23-aug-setup-5
# origin/run/23-aug-setup-6

# Fetch the latest changes from the remote
git fetch origin
BASE_BRANCH="develop"

BRANCH_PREFIX="run/23-aug-setup-"


get_branches() {
    git branch -r | grep "origin/$BRANCH_PREFIX" | sed 's/origin\///'
}

update_branches() {
    local branches=$(get_branches)
    for branch in $branches; do
        echo "Processing branch: $branch"
        git checkout $branch
        git pull origin $branch
        git merge origin/$BASE_BRANCH --no-edit -m "Merge $BASE_BRANCH into $branch"
        git push origin $branch
        echo "Finished processing $branch"
        echo "------------------------"
    done
    git checkout $BASE_BRANCH
    echo "All branches have been updated with $BASE_BRANCH"
}

print_diffs() {
    local branches=$(get_branches)
    for branch in $branches; do
        echo "----------------------------------------"
        echo "Difference between $branch and $BASE_BRANCH"
        git diff $BASE_BRANCH..$branch
    done
}

sanitize_tag() {
    echo "$1" | tr '/' '-'
}

build_images() {
    local branches=$(get_branches)
    for branch in $branches; do
        echo "Building and pushing image for branch: $branch"
        git checkout $branch
        sanitized_tag=$(sanitize_tag "$branch")
        make docker_push TAG=pascalwhoop-$sanitized_tag
        echo "Finished building and pushing image for $branch"
        echo "------------------------"
    done
    git checkout $BASE_BRANCH
}

submit_workflows() {
    echo "===================="
    echo "Submitting workflows"
    echo "===================="
    local branches=$(get_branches)
    for branch in $branches; do
        echo "Submitting workflow for $branch"
        echo "-------------------------------"
        sanitized_tag=$(sanitize_tag "$branch")
        argo submit -n argo-workflows --from wftmpl/matrix-lc-run \
          -p image=us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix \
          -p image_tag=pascalwhoop-$sanitized_tag \
          -p experiment=lc-baseline-run-$sanitized_tag \
          -p run_name=$sanitized_tag \
          -p neo4j_host=bolt://neo4j.neo4j.svc.cluster.local:7687 \
          -p mlflow_endpoint=http://mlflow-tracking.mlflow.svc.cluster.local:80 \
          -p openai_endpoint=https://api.openai.com/v1 \
          -p env=cloud \
          -l submit-from-ui=true \
          --entrypoint __default__
    done
}

run_all() {
    update_branches
    print_diffs
    build_images
}

case "$1" in
    update)
        update_branches
        ;;
    diff)
        print_diffs
        ;;
    build)
        build_images
        ;;
    submit)
        submit_workflows
        ;;
    *)
        run_all
        ;;
esac
