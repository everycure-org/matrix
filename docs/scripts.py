import os
import json
import yaml
import semver

CHANGELOG_FILE_DIR = "/Users/emil/github/dataminded/matrix/docs/src/releases/changelog_files/"


def list_files():
    files = os.listdir(CHANGELOG_FILE_DIR)
    return files


def filter_json_files(files):
    filtered_files = [file for file in files if file.endswith(".json")]
    return filtered_files


def semver_sort(files: list) -> list:
    sorted_files = sorted(files, key=semver.VersionInfo.parse)
    return sorted_files


def load_files(filepaths: list[str]):
    all_data = []
    for filepath in filepaths:
        with open(os.path.join(CHANGELOG_FILE_DIR, filepath), "r") as file:
            data = json.load(file)
            all_data.append(data)
    return all_data


def dump_to_yaml(files: list):
    yaml_data = yaml.dump(files, default_flow_style=False)
    return yaml_data


def save_yaml(yaml_data):
    with open(os.path.join(CHANGELOG_FILE_DIR, "releases_all-2.yaml"), "w") as file:
        file.write(yaml_data)


def main():
    files = list_files()
    filtered_files = filter_json_files(files)
    loaded_files = load_files(filtered_files)
    yaml_aggr = dump_to_yaml(loaded_files)
    save_yaml(yaml_aggr)
