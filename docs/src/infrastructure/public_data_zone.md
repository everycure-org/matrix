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

TBD, we will need to create a process for publishing data to the public zone. We wil start with the evidence.dev static website for now.

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
