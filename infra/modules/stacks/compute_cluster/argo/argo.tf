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

resource "helm_release" "argo_app_of_apps" {
  provider   = helm
  name       = "argocd-apps"
  depends_on = [helm_release.argo]
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argocd-apps"
  namespace  = var.namespace
  version    = "2.0.0"

  values = [
    yamlencode({
      "applications" : {
        "platform" : {
          "namespace" : var.namespace,
          "finalizers" : ["resources-finalizer.argocd.argoproj.io"],
          "project" : "default",
          "additionalLabels" : {
            "app.kubernetes.io/part-of" : join("-", ["argo", helm_release.argo.metadata[0].revision])
          }
          "source" : {
            "repoURL" : var.repo_url,
            "path" : var.repo_path,
            "targetRevision" : var.repo_revision,
          },
          "destination" : {
            "server" : "https://kubernetes.default.svc",
            "namespace" : var.namespace
          },
          "syncPolicy" : {
            "automated" : {
              "prune" : true,
              "selfHeal" : true
            }
          }
        }
      }
    })
  ]
}
