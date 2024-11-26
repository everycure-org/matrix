import functools
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Union

from matrix_cli.components.cache import memory
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential

console = Console()

# fix to avoid long CLI load times
if TYPE_CHECKING:
    from vertexai.generative_models import GenerationConfig, GenerativeModel


@functools.lru_cache
def load_vertex_model(model: str) -> "GenerativeModel":
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init()
    return GenerativeModel(model)


def get_git_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    ).stdout.strip()


@memory.cache()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120))
def invoke_model(prompt: str, model: str, generation_config: "GenerationConfig" = None) -> str:
    model_object = load_vertex_model(model)
    console.print(f"[bold green] Calling Gemini with a prompt of length: {len(prompt)} characters")
    response = model_object.generate_content(prompt, generation_config=generation_config).text
    console.print(f"[bold green] Response received. Total length: {len(response)} characters")
    return response


def get_markdown_contents(folder_path: Path | str) -> str:
    """Recursively reads and concatenates markdown files from a directory.

    Args:
        folder_path (Path | str): Path to the directory containing markdown files.

    Returns:
        str: Concatenated content of all markdown files, separated by delimiters.
    """
    folder_path = Path(folder_path)  # Convert to Path if string
    if not folder_path.exists():
        return ""

    all_content = ""
    for file in folder_path.glob("**/*.md"):
        with open(file, "r", encoding="utf-8") as f:
            all_content += "=" * 100 + "\n"
            all_content += f"{file.relative_to(folder_path)}\n"
            all_content += f.read() + "\n\n"
            all_content += "=" * 100 + "\n"

    return all_content


def run_command(
    command: Union[str, List[str]],
    cwd: str = None,
    check=True,
    log_before: bool = False,
    include_stderr: bool = False,
) -> Union[str, Tuple[str, str]]:
    if isinstance(command, str):
        # ensures we have an array but allow user to pass in string (for f-string style submission)
        command = shlex.split(command)

    if len(command) == 0:
        console.print("Command is empty... Exiting")
        exit(1)

    # enforcing everything is a string, sometimes we use posixpaths
    command = [str(item) for item in command]

    if log_before:
        console.print(" ".join(command))

    result = subprocess.run(command, check=check, capture_output=True, text=True, cwd=cwd)
    if include_stderr:
        return result.stdout.strip(), result.stderr.strip()
    else:
        return result.stdout.strip()
