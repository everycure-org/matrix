import os
import json
import yaml
import logging
from pathlib import Path


def locate_releases_path() -> Path:
    """Returns the absolute path of the changelog_files folder in this repo."""
    scripts_folder_path = Path(__file__).resolve().parent
    changelog_path = scripts_folder_path.parent / "src" / "releases" / "changelog_files"
    return changelog_path


def list_json_files(changelog_abs_path: str) -> list:
    """
    Returns a list of files with extension .json in the changelog files dir.
    """
    files = os.listdir(changelog_abs_path)
    json_files = filter_json_files(files)
    logging.info(f"Json files found in {changelog_abs_path} : {json_files}")
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


def load_files(filepaths: list[str], changelog_abs_path: str) -> list[dict]:
    """Loads a list of json files present in the changelog_files dir"""
    all_data = []
    for filepath in filepaths:
        with open(os.path.join(changelog_abs_path, filepath), "r") as file:
            data = json.load(file)
            all_data.append(data)
    return all_data


def create_semver_sortkey(filename: str) -> list[int]:
    version_str = filename.lstrip("v").split("-")[0]
    sort_key = [int(u) for u in version_str.split(".")]
    return sort_key


def dump_to_yaml(
    files: list[dict],
) -> str:
    yaml_data = yaml.dump(files, default_flow_style=False)
    return yaml_data


def save_yaml(yaml_data: str, changelog_abs_path: str) -> None:
    with open(os.path.join(changelog_abs_path, "releases_aggregated.yaml"), "w") as file:
        file.write(yaml_data)


def main() -> None:
    """Extracts json files from the changelog_files directory, aggregates them and saves as one yaml file"""
    changelog_abs_path = locate_releases_path()
    files = list_json_files(changelog_abs_path)
    if not files:
        raise ValueError(f"No json files found in {changelog_abs_path}")
    filtered_files = filter_json_files(files)
    loaded_files = load_files(filtered_files, changelog_abs_path)
    sorted_files = sorted(loaded_files, key=lambda x: create_semver_sortkey(x["Release Name"]))
    formatted_files = format_values(sorted_files)
    yaml_aggr = dump_to_yaml(formatted_files)
    save_yaml(yaml_aggr, changelog_abs_path)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    changelog_dir = current_dir.parent / "src" / "releases" / "changelog_files"

    print(current_dir)
    print(changelog_dir)
    main()
