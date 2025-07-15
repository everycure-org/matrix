import functools

import requests


@functools.cache
def _get_node_normalization_version(protocol_and_domain: str, openapi_path: str, source: str):
    nn_openapi_json_url = f"{protocol_and_domain}{openapi_path}"
    response = requests.get(nn_openapi_json_url, timeout=5)

    if not response.ok:
        return "nodenorm-test"

    json_response = response.json()
    version = json_response["info"]["version"]
    return f"nodenorm-{source.lower()}-{version}"


def get_node_normalization_settings(config: dict):
    if "protocol_and_domain" not in config or "get_normalized_nodes_path" not in config or "openapi_path" not in config:
        raise ValueError(f"Misconfigured Node Normalization settings")

    source = config["source"]
    protocol_and_domain = config["protocol_and_domain"]
    get_normalized_nodes_path = config["get_normalized_nodes_path"]
    openapi_path = config["openapi_path"]

    endpoint = f"{protocol_and_domain}{get_normalized_nodes_path}"
    version = _get_node_normalization_version(protocol_and_domain, openapi_path, source)

    return {"endpoint": endpoint, "version": version}
