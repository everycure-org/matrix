import functools
import os

import requests


@functools.cache
def get_node_normalization_version():
    """Function to get the NodeNormalization version."""
    nn_protocol_and_domain = os.getenv("NODE_NORM_PROTOCOL_AND_DOMAIN", "https://nodenormalization-sri.renci.org")
    if "renci" in nn_protocol_and_domain:
        source = "RENCI"
    elif "transltr" in nn_protocol_and_domain:
        source = "NCATS"
    else:
        source = "RENCI"
    nn_openapi_json_url = os.getenv("NODE_NORM_OPENAPI_JSON_URL", f"{nn_protocol_and_domain}/openapi.json")
    json_response = requests.get(f"{nn_openapi_json_url}").json()
    version = json_response["info"]["version"]
    return f"nodenorm-{source.lower()}-{version}"


@functools.cache
def get_node_normalization_get_normalized_nodes_url():
    """Function to get the NodeNormalization endpoint url."""
    nn_protocol_and_domain = os.getenv("NODE_NORM_PROTOCOL_AND_DOMAIN", "https://nodenormalization-sri.renci.org")
    nn_openapi_get_normalized_nodes_url = os.getenv(
        "NODE_NORM_OPENAPI_JSON_URL", f"{nn_protocol_and_domain}/1.5/get_normalized_nodes"
    )
    return nn_openapi_get_normalized_nodes_url


def get_node_normalization_parts(source: str):
    """Function to get the NodeNormalization url parts."""
    nn_protocol_and_domain = os.getenv("NODE_NORM_PROTOCOL_AND_DOMAIN", "https://nodenormalization-sri.renci.org")
    nn_openapi_json_url = os.getenv("NODE_NORM_OPENAPI_JSON_URL", f"{nn_protocol_and_domain}/openapi.json")
    return nn_openapi_json_url


NODE_NORM_ENDPOINT = get_node_normalization_get_normalized_nodes_url()

NODE_NORM_VERSION = get_node_normalization_version()
