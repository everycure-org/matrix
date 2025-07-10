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


@functools.cache
def _get_node_normalization_version(protocol_and_domain: str, openapi_path: str, source: str):
    nn_openapi_json_url = f"{protocol_and_domain}{openapi_path}"
    json_response = requests.get(f"{nn_openapi_json_url}").json()
    version = json_response["info"]["version"]
    return f"nodenorm-{source.lower()}-{version}"


def get_node_normalization_settings(config_name: str):
    if config_name not in NODE_NORMALIZER_CONFIGURATIONS:
        raise ValueError(f"Node normalization configuration must be one of: {NODE_NORMALIZER_CONFIGURATIONS.keys()}")

    config = NODE_NORMALIZER_CONFIGURATIONS[config_name]
    return {
        "endpoint": f"{config['protocol_and_domain']}{config['get_normalized_nodes_path']}",
        "version": _get_node_normalization_version(
            config["protocol_and_domain"], config["openapi_path"], config["source"]
        ),
    }
