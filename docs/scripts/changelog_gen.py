import os
import json
import yaml
import logging


def list_json_files() -> list:
    """
    Returns a list of files with extension .json in the current working directory.
    """
    cwd = os.getcwd()
    files = os.listdir(cwd)
    json_files = filter_json_files(files)
    logging.info(f"Json files found in {cwd} : {json_files}")
    return json_files


def filter_json_files(files: list) -> list:
    """Filters a list of filenames and retains files with .json extension"""
    filtered_files = [file for file in files if file.endswith(".json")]
    return filtered_files


def load_files(filepaths: list[str]) -> list[dict]:
    """Loads a list of json files present in the working directory"""
    all_data = []
    for filepath in filepaths:
        with open(os.path.join(os.getcwd(), filepath), "r") as file:
            data = json.load(file)
            all_data.append(data)
    return all_data


def dump_to_yaml(files: list[dict]) -> str:
    yaml_data = yaml.dump(files, default_flow_style=False)
    return yaml_data


def save_yaml(yaml_data):
    with open(os.path.join(os.getcwd(), "releases_aggregated.yaml"), "w") as file:
        file.write(yaml_data)


def main() -> None:
    """Extracts json files from the current working directory, aggregates them and saves as one yaml file"""
    files = list_json_files()
    if not files:
        raise ValueError(f"No json files found in {os.getcwd()}")
    filtered_files = filter_json_files(files)
    loaded_files = load_files(filtered_files)
    yaml_aggr = dump_to_yaml(loaded_files)
    save_yaml(yaml_aggr)


if __name__ == "__main__":
    main()
