# FUTURE: Why is this block required in the module?
terraform {
  required_providers {
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.7.0"
    }
  }
}

resource "kubernetes_namespace" "argo_ns" {
  metadata {
    name = var.namespace
  }
}

resource "kubernetes_secret" "argo_secret" {
  depends_on = [kubernetes_namespace.argo_ns]
  metadata {
    name      = "basic-auth"
    namespace = var.namespace
    labels = {
      "argocd.argoproj.io/secret-type" : "repository"
    }
  }
  data = {
    # type= base64encode("git")
    # url= base64encode(var.repo_url)
    # password= base64encode(var.repo_creds)
    type     = "git"
    url      = var.repo_url
    password = var.repo_creds
  }
  type = "Opaque"
}
resource "helm_release" "argo" {
  depends_on = [kubernetes_namespace.argo_ns, kubernetes_secret.argo_secret]
  name       = "argo"
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argo-cd"
  namespace  = var.namespace

  # pass through ssl to enable grpc/https for argocd CLI, see
  # https://argoproj.github.io/argo-cd/operator-manual/ingress/#kubernetesingress-nginx

  set {
    name  = "server.ingress.annotations.kubernetes.io/ingress.class"
    value = "nginx"
  }
  set {
    name  = "server.ingress.annotations.nginx.ingress.kubernetes.io/force-ssl-redirect"
    value = "true"
  }
  set {
    name  = "server.ingress.annotations.nginx.ingress.kubernetes.io/ssl-passthrough"
    value = "true"
  }
  set {
    name  = "configs.params.server\\.insecure"
    value = "true"
  }
  set {
    name  = "configs.params.server.basehref"
    value = "/"
  }
}

resource "kubernetes_manifest" "app_of_apps" {
  depends_on = [helm_release.argo, module.gke]
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
    helm:
      values: |
        global:
          environment: ${var.environment}
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