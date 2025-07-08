import functools

import requests

NODE_NORMALIZER_CONFIGURATIONS = {
    "RENCI": {
        "source": "RENCI",
        "protocol_and_domain": "https://nodenormalization-sri.renci.org",
        "get_normalized_nodes_path": "/1.5/get_normalized_nodes",
        "openapi_path": "/openapi.json",
    },
    "NCATS": {
        "source": "NCATS",
        "protocol_and_domain": "https://nodenorm.transltr.io",
        "get_normalized_nodes_path": "/1.5/get_normalized_nodes",
        "openapi_path": "/openapi.json",
    },
}


def _get_node_normalization_endpoint(config: dict[str, str]):
    return f"{config['protocol_and_domain']}{config['get_normalized_nodes_path']}"


@functools.cache
def _get_node_normalization_version(config: dict[str, str]):
    nn_openapi_json_url = f"{config['protocol_and_domain']}{config['openapi_path']}"
    json_response = requests.get(f"{nn_openapi_json_url}").json()
    version = json_response["info"]["version"]
    return f"nodenorm-{config['source'].lower()}-{version}"


def get_node_normalization_settings(config_name: str):
    if config_name not in NODE_NORMALIZER_CONFIGURATIONS:
        raise ValueError(f"Node normalization configuration must be one of: {NODE_NORMALIZER_CONFIGURATIONS.keys()}")

    config = NODE_NORMALIZER_CONFIGURATIONS[config_name]
    return {
        "endpoint": _get_node_normalization_endpoint(config),
        "version": _get_node_normalization_version(config),
    }
