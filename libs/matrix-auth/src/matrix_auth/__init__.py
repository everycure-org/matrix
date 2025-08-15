"""Matrix Authentication and Cloud Utilities Library.

This library provides authentication utilities for Google Cloud Platform (GCP).
"""

from .authentication import (
    get_service_account_creds,
    get_user_account_creds,
    request_new_iap_token,
)

__all__ = [
    "get_service_account_creds",
    "get_user_account_creds",
    "request_new_iap_token",
]
