#!/usr/bin/env bash

# This script is used for retry'ing to shut down the docker daemon. We're doing this because we 
# saw issues occur in CI where the docker daemon was supposedely not available but we are not able
# to reproduce it on the CI server when SSH'ing into it

# Configuration
MAX_RETRIES=5
RETRY_DELAY=5
COMPOSE_FILE="compose/docker-compose.yml"

# Function to check if Docker daemon is running with telemetry
check_docker_daemon() {
    if ! docker info > /dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Docker daemon not ready. Collecting telemetry..." >> "$LOG_FILE"
        
        # Check if Docker service is active
        systemctl status docker
        
        # Check for Docker socket
        ls -l /var/run/docker.socket
        ls -l /var/run/user/*/
        
        # Check Docker daemon logs
        echo "Last 10 lines of Docker daemon logs:"
        journalctl -u docker.service -n 10 --no-pager
        
        # Check system resources
        echo "System resource usage:"
        top -b -n 1 | head -n 20
        
        echo "Disk usage:"
        df -h
        
        return 1
    fi
    return 0
}

# Function to attempt Docker Compose shutdown
docker_compose_down() {
    docker compose -f "$COMPOSE_FILE" down
}

# Main execution
main() {
    echo "Starting Docker Compose shutdown process..."

    for attempt in $(seq 1 $MAX_RETRIES); do
        echo "Attempt $attempt of $MAX_RETRIES"

        if ! check_docker_daemon; then
            echo "Docker daemon is not running. Waiting before retry."
            sleep $RETRY_DELAY
            continue
        fi

        echo "Docker daemon is running. Proceeding with shutdown."
        if docker_compose_down; then
            echo "Docker Compose shutdown successful."
            return 0
        fi

        echo "Docker Compose shutdown failed."
        [ $attempt -lt $MAX_RETRIES ] && echo "Retrying in $RETRY_DELAY seconds..." && sleep $RETRY_DELAY
    done

    echo "Failed to shut down Docker Compose after $MAX_RETRIES attempts."
    return 1
}

# Run the main function
main
exit $?
