#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check and set up gcloud and kubectl
check_dependencies() {
    if ! command_exists gcloud; then
        echo "gcloud is not installed. Please install it first."
        exit 1
    fi

    if ! command_exists kubectl; then
        echo "kubectl is not installed. Installing it now..."
        gcloud components install kubectl
    fi

    # Ensure gcloud is authenticated and kubectl is configured
    if ! gcloud auth list --filter=status:ACTIVE --format="value(ACCOUNT)" &>/dev/null; then
        gcloud auth login
    fi
    echo "ensure we are authenticated with kubectl"
    gcloud container clusters get-credentials compute-cluster --project mtrx-hub-dev-3of --region us-central1
    # Check if kubectl is working by listing namespaces
    if ! kubectl get ns &>/dev/null; then
        echo "kubectl is not working. Exiting..."
        exit 1
    fi
}

# Function to build and push Docker image
build_push_docker() {
    make docker_push TAG=$USERNAME
    # Add your Docker build and push commands here
}

# Function to build Argo workflow template
build_argo_template() {
    echo "Building Argo workflow template..."
    # TODO duplicated image name reference from Makefile, should clean up
    IMAGE_NAME="us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
    .venv/bin/python ./src/matrix/argo.py generate-argo-config $IMAGE_NAME $RUN_NAME $USERNAME $NAMESPACE
}

# Function to create or verify namespace
ensure_namespace() {
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        kubectl create namespace "$NAMESPACE"
    fi
    echo "Using namespace: $NAMESPACE"
}

# Function to apply Argo template
apply_argo_template() {
    echo "Applying Argo workflow template..."
    # Add kubectl apply command for your Argo template
    kubectl apply -f templates/argo-workflow-template.yml -n $NAMESPACE
}

# Function to submit Argo workflow
submit_workflow() {

    #   -p openai_endpoint=https://api.openai.com/v1 \
    echo "Submitting Argo workflow..."
    JOB_NAME=$(argo submit --name $RUN_NAME -n $NAMESPACE --from wftmpl/matrix \
      -p run_name=$RUN_NAME \
      -l submit-from-ui=false \
      --entrypoint __default__ \
      -o json \
      | jq -r '.metadata.name')
    
    argo watch -n $NAMESPACE $JOB_NAME
}
get_experiment_name() {
    if [ -n "$RUN_NAME" ]; then
        echo "$RUN_NAME"
    else
        local branch_name=$(git rev-parse --abbrev-ref HEAD)
        echo "$branch_name" | tr -c '[:alnum:]-' '-' | sed 's/-$//'
    fi
}

# Main function
main() {
    set -xe
    check_dependencies
    build_push_docker
    build_argo_template
    # ensure_namespace NOTE: Currently executing in argo
    apply_argo_template
    submit_workflow
}
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --username)
            USERNAME="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="${2:-dev-$USER}"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --username <username>  Specify the username to use"
            echo "  --namespace <namespace>  Specify a custom namespace"
            echo "  --run-name <name>      Specify a custom run name"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

DEFAULT_USERNAME=$USER
DEFAULT_NAMESPACE="dev-$DEFAULT_USERNAME"
DEFAULT_RUN_NAME=$(get_experiment_name)
# After parsing, set defaults if not provided
USERNAME="${USERNAME:-$DEFAULT_USERNAME}"
NAMESPACE="${NAMESPACE:-$DEFAULT_NAMESPACE}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
echo "Running with the following parameters:"
echo "======================================"
echo "Username: $USERNAME"
echo "Namespace: $NAMESPACE"
echo "Run name: $(get_experiment_name)"

# Run the main function
main

