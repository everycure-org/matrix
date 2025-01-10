import os
import json
from typing import Iterable, Iterator

import yaml
import logging
from pathlib import Path


def locate_releases_path() -> Path:
    """Returns the absolute path of the changelog_files folder in this repo."""
    scripts_folder_path = Path(__file__).resolve().parent
    changelog_path = scripts_folder_path.parent / "src" / "releases" / "changelog_files"
    return changelog_path


def list_json_files(changelog_abs_path: Path) -> list:
    """
    Returns a list of files with extension .json in the changelog files dir.
    """
    json_files = list(Path(changelog_abs_path).glob("v*_info.json"))
    logging.debug(f"Json files found in '{changelog_abs_path}': {json_files}")
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


def parse_jsons(filepaths: Iterable[Path]) -> Iterator[dict]:
    """Parse the contents of the files in `filepaths` as Json objects."""
    for filepath in filepaths:
        yield json.loads(filepath.read_text())


def create_semver_sortkey(filename: str) -> list[int]:
    version_str = filename.lstrip("v").split("-")[0]
    sort_key = [int(u) for u in version_str.split(".")]
    return sort_key


def sort_releases(releases: Iterable[dict]) -> list[dict]:
    sorted_list = sorted(releases, key=lambda x: create_semver_sortkey(x["Release Name"]))
    return sorted_list


def dump_to_yaml(
    files: list[dict],
) -> str:
    yaml_data = yaml.dump(files, default_flow_style=False, sort_keys=False)
    return yaml_data


def save_yaml(yaml_data: str, changelog_abs_path: Path) -> None:
    with open(os.path.join(changelog_abs_path, "releases_aggregated.yaml"), "w") as file:
        file.write(yaml_data)


def main() -> None:
    """Extracts json files from the changelog_files directory, aggregates them and saves as one yaml file"""
    changelog_abs_path = locate_releases_path()
    files = list_json_files(changelog_abs_path)
    if not files:
        raise ValueError(f"No json files found in {changelog_abs_path}")
    loaded_files = parse_jsons(files)
    sorted_files = sort_releases(loaded_files)
    formatted_files = format_values(sorted_files)
    yaml_aggr = dump_to_yaml(formatted_files)
    save_yaml(yaml_aggr, changelog_abs_path)


if __name__ == "__main__":
    main()
