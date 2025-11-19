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
      "argocd.argoproj.io/secret-type" : "repo-creds"
    }
  }
  data = {
    # type= base64encode("git")
    # url= base64encode(var.repo_url)
    # password= base64encode(var.repo_creds)
    type     = "git"
    url      = "https://github.com/everycure-org/"
    password = var.repo_creds
  }
  type = "Opaque"
}

resource "helm_release" "argo" {
  depends_on    = [kubernetes_namespace.argo_ns, kubernetes_secret.argo_secret]
  name          = "argo"
  repository    = "https://argoproj.github.io/argo-helm"
  chart         = "argo-cd"
  namespace     = var.namespace
  timeout       = 600
  atomic        = true
  recreate_pods = true
  version       = "8.3.1" # Pin to a specific version to avoid breaking changes

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

  # Disable Git submodule initialization
  set {
    name  = "configs.params.reposerver\\.enable\\.git\\.submodule"
    value = "false"
  }

  # Configure ArgoCD components to run on management nodes
  # Application Controller
  set {
    name  = "controller.nodeSelector.workload-type"
    value = "management"
  }
  set {
    name  = "controller.tolerations[0].key"
    value = "workload-type"
  }
  set {
    name  = "controller.tolerations[0].operator"
    value = "Equal"
  }
  set {
    name  = "controller.tolerations[0].value"
    value = "management"
  }
  set {
    name  = "controller.tolerations[0].effect"
    value = "NoSchedule"
  }

  # Server
  set {
    name  = "server.nodeSelector.workload-type"
    value = "management"
  }
  set {
    name  = "server.tolerations[0].key"
    value = "workload-type"
  }
  set {
    name  = "server.tolerations[0].operator"
    value = "Equal"
  }
  set {
    name  = "server.tolerations[0].value"
    value = "management"
  }
  set {
    name  = "server.tolerations[0].effect"
    value = "NoSchedule"
  }

  # Repo Server
  set {
    name  = "repoServer.nodeSelector.workload-type"
    value = "management"
  }
  set {
    name  = "repoServer.tolerations[0].key"
    value = "workload-type"
  }
  set {
    name  = "repoServer.tolerations[0].operator"
    value = "Equal"
  }
  set {
    name  = "repoServer.tolerations[0].value"
    value = "management"
  }
  set {
    name  = "repoServer.tolerations[0].effect"
    value = "NoSchedule"
  }

  # ApplicationSet Controller
  set {
    name  = "applicationSet.nodeSelector.workload-type"
    value = "management"
  }
  set {
    name  = "applicationSet.tolerations[0].key"
    value = "workload-type"
  }
  set {
    name  = "applicationSet.tolerations[0].operator"
    value = "Equal"
  }
  set {
    name  = "applicationSet.tolerations[0].value"
    value = "management"
  }
  set {
    name  = "applicationSet.tolerations[0].effect"
    value = "NoSchedule"
  }

  # Redis
  set {
    name  = "redis.nodeSelector.workload-type"
    value = "management"
  }
  set {
    name  = "redis.tolerations[0].key"
    value = "workload-type"
  }
  set {
    name  = "redis.tolerations[0].operator"
    value = "Equal"
  }
  set {
    name  = "redis.tolerations[0].value"
    value = "management"
  }
  set {
    name  = "redis.tolerations[0].effect"
    value = "NoSchedule"
  }

  # Dex Server
  set {
    name  = "dex.nodeSelector.workload-type"
    value = "management"
  }
  set {
    name  = "dex.tolerations[0].key"
    value = "workload-type"
  }
  set {
    name  = "dex.tolerations[0].operator"
    value = "Equal"
  }
  set {
    name  = "dex.tolerations[0].value"
    value = "management"
  }
  set {
    name  = "dex.tolerations[0].effect"
    value = "NoSchedule"
  }

  # Notifications Controller
  set {
    name  = "notifications.nodeSelector.workload-type"
    value = "management"
  }
  set {
    name  = "notifications.tolerations[0].key"
    value = "workload-type"
  }
  set {
    name  = "notifications.tolerations[0].operator"
    value = "Equal"
  }
  set {
    name  = "notifications.tolerations[0].value"
    value = "management"
  }
  set {
    name  = "notifications.tolerations[0].effect"
    value = "NoSchedule"
  }
}

resource "kubernetes_manifest" "app_of_apps" {
  depends_on = [helm_release.argo]

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
  project: default
  source:
    path: ${var.repo_path}/app-of-apps
    repoURL: ${var.repo_url}
    targetRevision: change_claude_to_anthrophic_in_litellm
    helm:
      parameters:
      - name: spec.source.targetRevision
        value:  change_claude_to_anthrophic_in_litellm
      - name: spec.source.environment
        value:  ${var.environment}
      - name: spec.source.project_id
        value: ${var.project_id}
      - name: spec.source.bucketname
        value: ${var.bucket_name}
      - name: spec.source.aip_oauth_client_id
        value: ${var.aip_oauth_client_id}
  syncPolicy:
    syncOptions:
      - CreateNamespace=true
    automated:
      prune: true
      allowEmpty: true
YAML
  )
}