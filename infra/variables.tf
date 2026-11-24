variable "kube_context" {
  description = "Kubernetes context name (e.g., minikube or kind)"
  type        = string
  default     = "minikube"
}

variable "namespace_name" {
  description = "Target Kubernetes namespace for MLOps workloads"
  type        = string
  default     = "mlops-k8s"
}

variable "project_label" {
  description = "Project identifier label"
  type        = string
  default     = "k8s-mlops-project"
}
