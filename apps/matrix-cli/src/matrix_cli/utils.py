import subprocess
from pathlib import Path

from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential
from vertexai.generative_models import (
    GenerationConfig,
)

from matrix_cli.cache import memory

console = Console()


def get_git_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    ).stdout.strip()


@memory.cache()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120))
def invoke_model(prompt: str, model: str, generation_config: GenerationConfig = None) -> str:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init()
    model_object = GenerativeModel(model)
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
