import functools
import os

import requests


@functools.cache
def get_node_normalization_version():
    """Function to get the NodeNormalization version."""
    source = os.getenv("NODE_NORM_SOURCE", "RENCI")
    protocol_and_domain, get_normalized_nodes_path, openapi_path = get_node_normalization_parts(source)
    json_response = requests.get(f"{protocol_and_domain}{openapi_path}").json()
    version = json_response["info"]["version"]
    return f"nodenorm-{source.lower()}-{version}"


@functools.cache
def get_node_normalization_get_normalized_nodes_url():
    """Function to get the NodeNormalization endpoint url."""
    source = os.getenv("NODE_NORM_SOURCE", "RENCI")
    protocol_and_domain, get_normalized_nodes_path, openapi_path = get_node_normalization_parts(source)
    return f"{protocol_and_domain}{get_normalized_nodes_path}"


def get_node_normalization_parts(source: str):
    """Function to get the NodeNormalization url parts."""
    match source:
        case "NCATS":
            protocol_and_domain = "https://nodenorm.transltr.io"
        case "RENCI":
            protocol_and_domain = "https://nodenormalization-sri.renci.org"
        case _:
            protocol_and_domain = "https://nodenormalization-sri.renci.org"
    get_normalized_nodes_path = "/1.5/get_normalized_nodes"
    openapi_path = "/openapi.json"
    return protocol_and_domain, get_normalized_nodes_path, openapi_path


NODE_NORM_ENPOINT = get_node_normalization_get_normalized_nodes_url()

NODE_NORM_VERSION = get_node_normalization_version()
