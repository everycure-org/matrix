---
title: Set Up Docker for Large Data Processing
---

# Optimizing Docker for Large Data Processing

Now that you have successfully set up Docker for basic pipeline execution, this guide will help you optimize your Docker environment for handling large datasets and real-world data processing scenarios. The Matrix pipeline processes substantial amounts of data, requiring careful resource management and configuration tuning.

## Overview

When working with real data in the Matrix pipeline, you'll encounter significantly higher resource demands compared to the test environment. This guide covers:

- **Memory Optimization**: Configuring Docker and services for large data processing
- **Neo4j Tuning**: Optimizing the graph database for heavy workloads
- **Storage Management**: Handling large datasets and intermediate files
- **Performance Monitoring**: Tools and techniques for monitoring resource usage

!!! important "Resource Requirements"
    For large data processing, ensure your system has:
    - **Minimum 16GB RAM** (32GB+ recommended)
    - **50GB+ free disk space** for data and containers
    - **Multi-core CPU** for parallel processing
    - **SSD storage** for better I/O performance

## Docker Memory Configuration

### System-Level Docker Settings

Before running large data workloads, optimize your Docker settings:

#### macOS/Windows Docker Desktop
1. Open Docker Desktop → Settings → Resources
2. **Memory**: Set to at least 16GB (or 70% of available RAM)
3. **Swap**: Increase to 4-8GB for additional memory buffer
4. **Disk Image Size**: Increase to 100GB+ for large datasets
5. **CPU**: Allocate at least 4 cores

#### Linux Docker
```bash
# Create or edit /etc/docker/daemon.json
{
  "default-ulimits": {
    "nofile": {
      "Hard": 65536,
      "Name": "nofile",
      "Soft": 65536
    }
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ]
}

# Restart Docker daemon
sudo systemctl restart docker
```

### Container Memory Limits

Create a custom `docker-compose.override.yml` file for large data processing:

```yaml
# docker-compose.override.yml
version: '3.8'

services:
  neo4j:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 6G
    environment:
      # Neo4j memory configuration (70% of container memory)
      - NEO4J_dbms_memory_heap_initial__size=5G
      - NEO4J_dbms_memory_heap_max__size=6G
      - NEO4J_dbms_memory_pagecache_size=2G
      # Additional performance settings
      - NEO4J_dbms_memory_off_heap_max_size=1G
      - NEO4J_dbms_memory_off_heap_enabled=true
      - NEO4J_dbms_connector_bolt_thread_pool_max_size=100
      - NEO4J_dbms_connector_http_thread_pool_max_size=100

  mlflow:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  mockserver:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

## Neo4j Optimization for Large Data

### Memory Configuration

Neo4j is the most memory-intensive component. Optimize it based on your data size:

#### Small Datasets (< 1M nodes)
```yaml
environment:
  - NEO4J_dbms_memory_heap_initial__size=2G
  - NEO4J_dbms_memory_heap_max__size=3G
  - NEO4J_dbms_memory_pagecache_size=1G
```

#### Medium Datasets (1M - 10M nodes)
```yaml
environment:
  - NEO4J_dbms_memory_heap_initial__size=6G
  - NEO4J_dbms_memory_heap_max__size=8G
  - NEO4J_dbms_memory_pagecache_size=4G
```

#### Large Datasets (> 10M nodes)
```yaml
environment:
  - NEO4J_dbms_memory_heap_initial__size=12G
  - NEO4J_dbms_memory_heap_max__size=16G
  - NEO4J_dbms_memory_pagecache_size=8G
```

### Performance Tuning

Add these environment variables to your Neo4j service:

```yaml
environment:
  # Existing settings...
  
  # Performance optimizations
  - NEO4J_dbms_memory_off_heap_max_size=2G
  - NEO4J_dbms_memory_off_heap_enabled=true
  - NEO4J_dbms_connector_bolt_thread_pool_max_size=200
  - NEO4J_dbms_connector_http_thread_pool_max_size=200
  
  # Transaction management
  - NEO4J_dbms_tx_log_rotation_retention_policy=1G
  - NEO4J_dbms_tx_log_rotation_size=256M
  
  # Query optimization
  - NEO4J_dbms_query_cache_size=1000
  - NEO4J_dbms_query_cache_hits_threshold=100
  
  # Logging optimization
  - NEO4J_db_logs_query_enabled=OFF
  - NEO4J_db_logs_query_parameter_logging_enabled=OFF
```

## Storage Optimization

### Volume Configuration

Optimize storage for large datasets:

```yaml
services:
  neo4j:
    volumes:
      # Use named volumes for better performance
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      # Mount host directories for large datasets
      - ./large_datasets:/var/lib/neo4j/import/large_datasets:ro

volumes:
  neo4j_data:
    driver: local
  neo4j_logs:
    driver: local
  neo4j_import:
    driver: local
```

### Data Directory Structure

Organize your data directories for optimal performance:

```bash
# Create optimized directory structure
mkdir -p ./large_datasets/{raw,processed,temp}
mkdir -p ./neo4j_data/{data,logs,import,backup}

# Set appropriate permissions
chmod 755 ./large_datasets
chmod 755 ./neo4j_data
```

## Monitoring and Troubleshooting

### Resource Monitoring

Monitor your Docker containers during large data processing:

```bash
# Monitor resource usage
docker stats

# Monitor specific container
docker stats neo4j mlflow

# Check container logs
docker logs -f neo4j
docker logs -f mlflow
```

### Common Issues and Solutions

#### Out of Memory (OOM) Errors

**Symptoms**: Containers being killed with OOM errors

**Solutions**:
```bash
# Increase Docker memory limit
# Check current memory usage
docker stats

# Restart with more memory
docker-compose down
# Edit docker-compose.override.yml with higher limits
docker-compose up -d
```

#### Neo4j Memory Issues

**Symptoms**: Neo4j container stuck at "Waiting for neo4j to be ready..."

**Solutions**:
```yaml
# Reduce Neo4j heap size to 70% of container memory
environment:
  - NEO4J_dbms_memory_heap_initial__size=5G  # 70% of 8G container
  - NEO4J_dbms_memory_heap_max__size=6G      # 70% of 8G container
```

#### Disk Space Issues

**Symptoms**: Build failures with "No space left on device"

**Solutions**:
```bash
# Clean up Docker system
docker system prune -a

# Clean up unused volumes
docker volume prune

# Check disk usage
df -h
du -sh ./neo4j_data/*
```
## Scaling Considerations

### For Very Large Datasets

If you're processing datasets with >100M nodes:

1. **Consider Cloud Deployment**: Use GCP or AWS for large-scale processing
2. **Distributed Processing**: Use Spark cluster instead of local processing
3. **Data Partitioning**: Split data into manageable chunks
4. **Incremental Processing**: Process data incrementally rather than all at once

### Hardware Recommendations

For optimal performance with large data:

- **RAM**: 64GB+ for datasets >10M nodes
- **Storage**: NVMe SSD with 1TB+ capacity
- **CPU**: 8+ cores for parallel processing
- **Network**: High-speed connection for cloud data access


