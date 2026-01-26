def parse_release_version_from_matrix_tag(release_version: str) -> str:
    return release_version.lstrip("v").rstrip("-matrix")
