---
title: Public Data Zone
---

# Public Data Zone

This document provides an overview of our public data zone infrastructure, which uses Google Cloud Storage (GCS) for data and static website distribution. The setup provides a secure, scalable, and performant way to serve public data and static websites.

## Architecture Overview

Our public data zone consists of a Google Cloud Storage bucket fronted by Cloud CDN and secured with HTTPS. The architecture is designed to provide:

- Secure public access to data
- High-performance content delivery
- Version control for data
- Static website hosting capabilities

![Public Data Zone Architecture](https://lh3.googleusercontent.com/xS0HnRmmCUscXcDJn73duq9h9pU2v0B-45QKahOs1ZmyNNkb0TcslcAyhDHcXeuniDbePB2n25Dn=s2048-w2048-rw-lo)

## Components

### Storage Bucket

The configuration for the public data zone bucket is as follows:

- **Bucket Name**: data.dev.everycure.org
- **Features**:
  - Versioning is enabled
  - Uniform bucket level access is set
  - Website hosting is configured with index.html and 404.html files (which are not present by default)
  - CORS is enabled for cross-origin requests
  - Public read access is granted

### Content Delivery

We use Google Cloud CDN to ensure fast content delivery:

- Global edge caching
- Automatic SSL/TLS certificate management
- HTTP to HTTPS redirection
- Load balancing for high availability

### Security

The public data zone implements several security measures:

1. **HTTPS Enforcement**
   - Google-managed SSL certificates
   - Automatic HTTP to HTTPS redirection
   - Secure content delivery

2. **Access Control**
   - Public read-only access
   - Uniform bucket level access
   - Version control for data integrity

3. **CORS Configuration**

```yaml
cors:
  origin: ["*"]
  methods: ["GET", "HEAD", "OPTIONS"]
  response_headers: ["*"]
  max_age: 3600
```

The CORS configuration allows all origins, methods, and headers, and sets a maximum age of 1 hour for the cache. This lets others work with our data easily from 3rd party websites. 

## Usage

### Accessing the Public Data

Data in the public zone can be accessed through:

1. **HTTPS Endpoint**
   ```
   https://data.dev.everycure.org/<path-to-data>
   ```

2. **Direct GCS Access**
   ```
   gs://data.dev.everycure.org/<path-to-data>
   ```

### Publishing Data

We use the `upload_release_to_public_bucket.sh` script to publish data from our internal
storage to the public data zone. This script handles downloading data from our
development bucket, compressing it, and uploading it to the public bucket. This is a very
rudimentary script and the result is a single tar.gz file for each version. As we zip up
parquet files, tarball files are an acceptable format, but future versions will likely 
try to create more accessible releases. For now, the accessible approach to share our data
is via bigquery, neo4j and kg dashboard.

#### Script Features

- Downloads specific version data from internal buckets
- Compresses data using parallel gzip (pigz) for efficiency
- Uploads the compressed data to the public bucket in a versioned directory structure
- Includes error handling and dependency checks

#### Prerequisites

- `gcloud` CLI tool installed and authenticated
- `pigz` parallel compression tool installed
- Appropriate permissions to access both source and destination buckets

#### Usage

```bash
./scripts/upload_release_to_public_bucket.sh [VERSION]
```

Where `[VERSION]` is the version tag in the format `vX.Y.Z` (e.g., `v0.4.4`). If not specified, the script defaults to `v0.4.4`.

#### Example

To publish data for version `v0.4.4`:

```bash
./scripts/upload_release_to_public_bucket.sh v0.4.4
```

This will:
1. Download the data from `gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/v0.4.4/datasets/release/prm/bigquery_edges/` and `bigquery_nodes/`
2. Compress the data into a `data.tar.gz` file
3. Upload it to `gs://data.dev.everycure.org/versions/v0.4.4/data/data.tar.gz`

The data will then be accessible via:
```
https://data.dev.everycure.org/versions/v0.4.4/data/data.tar.gz
```

### Hosting Static Websites

The bucket can host static websites with these features:

- Automatic index.html serving
- Custom 404 error pages
- CDN caching for performance
- Automatic SSL certificate management

## Environment Support

The infrastructure code supports multiple environments. The environment can be passed in as a variable, same as the Cloud CDN zone that is used to set the DNS records.

## Infrastructure Management

The entire infrastructure is managed through Terraform in the `data_release_zone` stack:

```hcl
module "data_release_zone" {
  source                = "../../../modules/stacks/data_release_zone"
  project_id            = var.project_id
  region                = var.default_region
  dns_managed_zone_name = module.dns.dns_zone
  environment           = "dev"
}
```

- [Google Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Cloud CDN Overview](https://cloud.google.com/cdn/docs/overview)
- [Managing SSL Certificates](https://cloud.google.com/load-balancing/docs/ssl-certificates)
