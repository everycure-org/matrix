"""Matrix Authentication and Cloud Utilities Library.

This library provides authentication utilities for Google Cloud Platform (GCP)
and Kubernetes cluster management functionality.
"""

from .authentication import (
    get_service_account_creds,
    get_user_account_creds,
    request_new_iap_token,
)
from .kubernetes import (
    apply,
    can_talk_to_kubernetes,
    create_namespace,
    namespace_exists,
)

__all__ = [
    "get_service_account_creds",
    "get_user_account_creds",
    "request_new_iap_token",
    "can_talk_to_kubernetes",
    "namespace_exists",
    "create_namespace",
    "apply",
]
