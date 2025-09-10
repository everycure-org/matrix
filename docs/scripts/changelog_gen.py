"""Aggregate all the individual changelog files per release into a single YAML
that can immediately be referenced by the Releases page on the documentation
website."""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import yaml


def locate_releases_path() -> Path:
    """Returns the absolute path of the changelog_files folder in this repo."""
    scripts_folder_path = Path(__file__).resolve().parent
    changelog_path = scripts_folder_path.parent / "src" / "releases" / "changelog_files"
    return changelog_path


def list_json_files(changelog_abs_path: Path, filename_pattern: str = "v*_info.json") -> Iterable[Path]:
    """
    Returns a list of files matching the `filename_pattern` in the changelog files dir.
    """
    json_files = Path(changelog_abs_path).glob(filename_pattern)
    return json_files


def is_markdown_link(text):
    """Check if string has the shape [*](*)"""
    pattern = r"^\[.*?\]\(.*?\)$"
    return bool(re.match(pattern, text))


def format_values(loaded_files):
    link_tmpl = "[Link]({value})"

    link_columns = ["BigQuery", "KG dashboard", "MLFlow", "Code", "NodeNorm Endpoint"]
    already_formatted_link_columns = ["NodeNorm Endpoint"]

    for file in loaded_files:
        for key, value in file.items():
            if key in already_formatted_link_columns and is_markdown_link(value):
                file[key] = value
            elif key in link_columns:
                file[key] = link_tmpl.format(value=value)
    return loaded_files


def parse_jsons(filepaths: Iterable[Path]) -> Iterator[dict]:
    """Parse the contents of the files in `filepaths` as Json objects."""
    for filepath in filepaths:
        yield json.loads(filepath.read_text())


def create_semver_sortkey(release_name: str) -> list[int]:
    version_str = release_name.lstrip("v").split("-")[0]
    sort_key = [int(u) for u in version_str.split(".")]
    return sort_key


def sort_releases(releases: Iterable[dict]) -> list[dict]:
    sorted_list = sorted(releases, key=lambda x: create_semver_sortkey(x["Release Name"]), reverse=True)
    return sorted_list


def dump_to_yaml(
    files: list[dict],
) -> str:
    yaml_data = yaml.dump(files, default_flow_style=False, sort_keys=False)
    return yaml_data


def save_yaml(yaml_data: str, changelog_abs_path: Path) -> None:
    (changelog_abs_path / "releases_aggregated.yaml").write_text(yaml_data)


def stringify_datasources(files: List[Dict]) -> List[Dict]:
    """Formats the dictionary to be rendered correctly using read_yaml of mkdocs."""
    for release_data in files:
        release_data["Datasets"] = "<br/>".join(f"‣ {k}: `{v}`" for k, v in release_data["Datasets"].items())
    return files


def main() -> None:
    """Extracts json files from the changelog_files directory, aggregates them and saves as one yaml file"""
    changelog_abs_path = locate_releases_path()
    files = list_json_files(changelog_abs_path)
    releases = parse_jsons(files)
    sorted_releases = sort_releases(releases)
    if not sorted_releases:
        raise ValueError(f"No release info found in {changelog_abs_path}")
    formatted_files = format_values(sorted_releases)
    formatted_files = stringify_datasources(formatted_files)
    yaml_aggr = dump_to_yaml(formatted_files)
    save_yaml(yaml_aggr, changelog_abs_path)


if __name__ == "__main__":
    main()
