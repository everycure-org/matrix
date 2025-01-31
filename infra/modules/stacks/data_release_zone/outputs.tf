output "website_url" {
  description = "The URI of the Evidence.dev website bucket"
  value       = module.public_data_bucket.website_url
}

# output "evidence_dev_website_url" {
#   description = "The URL where the Evidence.dev website is accessible"
#   value       = module.evidence_dev_website.website_url
# }
# 
# output "evidence_dev_bucket_name" {
#   description = "The name of the bucket hosting the Evidence.dev website"
#   value       = module.evidence_dev_website.website_bucket_name
# }
# 
# 
# output "evidence_dev_access_logs_bucket" {
#   description = "The name of the access logs bucket for the Evidence.dev website"
#   value       = module.evidence_dev_website.access_logs_bucket_name
# } 