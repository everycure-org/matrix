# FUTURE: Why is this block required in the module?
terraform {
  required_providers {
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.7.0"
    }
  }
}

resource "kubectl_manifest" "argo_ns" {
  yaml_body = <<YAML
apiVersion: v1
kind: Namespace
metadata:
  name:  ${var.namespace}
  YAML
}

resource "kubectl_manifest" "argo_creds" {
  yaml_body = <<YAML
apiVersion: v1
kind: Secret
metadata:
  name: basic-auth
  namespace: ${var.namespace}
  labels:
    "argocd.argoproj.io/secret-type" : "repository"
data:
   type: ${base64encode("git")}
   url: ${base64encode(var.repo_url)}
   password: ${base64encode(var.repo_creds)}
type: Opaque
  YAML
}

resource "helm_release" "argo" {
  name       = "argo"
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argo-cd"
  namespace  = var.namespace

  # pass through ssl to enable grpc/https for argocd CLI, see
  # https://argoproj.github.io/argo-cd/operator-manual/ingress/#kubernetesingress-nginx

  values = [
    # config that goes into the declarative config of argocd
    # https://argo-cd.readthedocs.io/en/stable/operator-manual/declarative-setup/#repositories
    yamlencode({
      "server.ingress.annotations" : {
        "kubernetes.io/ingress.class" : "nginx",
        "nginx.ingress.kubernetes.io/force-ssl-redirect" : "true",
        "nginx.ingress.kubernetes.io/ssl-passthrough" : "true",
      }
    })
  ]
}

resource "kubernetes_manifest" "app_of_apps" {
  depends_on = [helm_release.argo]
  # TODO try and remove 
  field_manager {
    force_conflicts = true
  }
  manifest = yamldecode(
<<YAML
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: app-of-apps
  namespace: argocd
spec:
  destination:
    namespace: argocd
    server: "https://kubernetes.default.svc"
  source:
    path: ${var.repo_path}/app-of-apps
    repoURL: ${var.repo_url}
    targetRevision: ${var.repo_revision}
  project: default
  syncPolicy:
    syncOptions:
      - CreateNamespace=true
    automated:
      prune: true
      allowEmpty: true
YAML
)
}