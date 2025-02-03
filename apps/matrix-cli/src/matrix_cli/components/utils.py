import functools
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List

import questionary
import semver
import typer
from google.auth import default
from matrix_cli.components.cache import memory
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential

console = Console()

# fix to avoid long CLI load times
if TYPE_CHECKING:
    from vertexai.generative_models import GenerationConfig, GenerativeModel


@functools.lru_cache
def load_vertex_model(model: str, creds) -> "GenerativeModel":
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(credentials=creds)
    return GenerativeModel(model)


def get_git_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    ).stdout.strip()


@memory.cache()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120))
def invoke_model(prompt: str, model: str, generation_config: "GenerationConfig" = None) -> str:
    credentials, project_id = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    model_object = load_vertex_model(model, credentials)
    console.print(f"[bold green] Calling Gemini with a prompt of length: {len(prompt)} characters")
    response = model_object.generate_content(prompt, generation_config=generation_config).text
    console.print(f"[bold green] Response received. Total length: {len(response)} characters")
    return response


def get_markdown_contents(folder_path: Path | str) -> str:
    """Iteratively reads and concatenates markdown files from a directory.

    Args:
        folder_path (Path | str): Path to the directory containing markdown files.

    Returns:
        str: Concatenated content of all markdown files, separated by delimiters.
    """
    folder_path = Path(folder_path)  # Convert to Path if string
    if not folder_path.exists():
        return ""
    line = "=" * 100
    all_content = [line]
    for file in folder_path.glob("**/*.md"):
        all_content.append(str(file.relative_to(folder_path)))
        all_content.append(file.read_text(encoding="utf-8"))
        all_content.append("\n" + line)
    return "\n".join(all_content)


def run_command(command: List[str], cwd: str = None) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running command {' '.join(command)}: {e.stderr}", err=True)
        raise


def ask_for_release():
    """Prompts the user to select an existing release from the list of releases."""
    console.print("[green]What was the last release that we should use as a starting point?")
    return questionary.select(
        "Select existing release from the list below:",
        choices=get_releases(),
    ).ask()


def get_latest_minor_release() -> str:
    releases_list = get_releases()
    # Map the case where the release is not in the semver compliant format x.y.z
    mapper = {"v0.1": "v0.1.0", "v0.2": "v0.2.0"}
    mapped_releases = [mapper.get(release, release) for release in releases_list]
    # Store original-to-mapped version mapping
    original_to_mapped = {mapper.get(release, release): release for release in releases_list}
    # Remove 'v' prefix and parse versions
    try:
        parsed_versions = [semver.Version.parse(v.lstrip("v")) for v in mapped_releases]
    except ValueError as e:
        raise ValueError(f"[red]Error parsing versions: {e}")

    # Get the latest minor version
    latest_minor = max(parsed_versions, key=lambda v: (v.major, v.minor)).minor
    # Get the earlist patch within the latest minor
    latest_minor_release = min([v for v in parsed_versions if v.minor == latest_minor], key=lambda v: v.patch)
    return original_to_mapped[f"v{latest_minor_release}"]


def get_releases() -> List[str]:
    # Sort releases by version using semantic versioning
    # Remark: repo should be fully checked out for this to work!
    return run_command(["gh", "release", "list", "--json", "tagName", "--jq", ".[].tagName"], cwd=get_git_root()).split(
        "\n"
    )
