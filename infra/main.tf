resource "kubernetes_namespace" "mlops" {
  metadata {
    name = var.namespace_name
    labels = {
      project = var.project_label
      managed = "terraform"
    }
  }
}

resource "kubernetes_config_map" "infra_info" {
  metadata {
    name      = "infra-metadata"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }

  data = {
    cluster_driver = "docker"
    workflow_stage = "terraform-provisioned"
  }
}
