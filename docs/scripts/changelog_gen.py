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


def format_values(loaded_files):
    link_tmpl = "[Link]({value})"
    code_tmpl = "`{value}`"
    for file in loaded_files:
        for key, value in file.items():
            if key.lower().endswith("link"):
                file[key] = link_tmpl.format(value=value)
            elif key.lower().endswith(("encoder", "estimator")):
                file[key] = code_tmpl.format(value=value)
    return loaded_files


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


def create_semver_sortkey(filename: str) -> list:
    version_str = filename.lstrip("v").split("-")[0]
    sort_key = [int(u) for u in version_str.split(".")]
    return sort_key


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
    sorted_files = sorted(loaded_files, key=create_semver_sortkey)
    formatted_files = format_values(sorted_files)
    yaml_aggr = dump_to_yaml(formatted_files)
    save_yaml(yaml_aggr)


if __name__ == "__main__":
    main()
