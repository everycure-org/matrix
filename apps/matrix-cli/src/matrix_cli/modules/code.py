# NOTE: This module was partially generated using AI assistance.

import os
import subprocess
from pathlib import Path

import typer
from rich import print
from rich.markdown import Markdown

app = typer.Typer(help="Code-related utility commands")


def get_code_diff(since: str) -> str:
    """Get code differences since a specific git reference.

    Args:
        since: Git reference (SHA, tag, or branch) to diff from

    Returns:
        str: Formatted diff output
    """
    script_path = Path(os.getcwd()) / "tools" / "diff_between_code.sh"

    try:
        result = subprocess.run(["bash", "-c", str(script_path), since], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise typer.BadParameter(f"Failed to get diff: {e.stderr}")


@app.command()
def catchup(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to diff from"),
):
    """Show code changes since a specific git reference."""
    try:
        diff_output = get_code_diff(since)
        print(Markdown(f"```diff\n{diff_output}\n```"))
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def summarize(
    since: str = typer.Argument(..., help="Git reference (SHA, tag, branch) to summarize from"),
    model: str = typer.Option("gemini-pro", help="Model to use for summarization"),
):
    """Generate an AI summary of code changes since a specific git reference."""

    import vertexai
    from vertexai.generative_models import GenerativeModel

    try:
        typer.echo(f"Loading diff...: {model}")
        diff_output = get_code_diff(since)
        typer.echo("Diff loaded! Loading AI model...")
        # Configure Gemini
        vertexai.init()
        model = GenerativeModel("gemini-1.5-pro-002")

        # response = model.generate_content(
        #     "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
        # )
        #
        # print(response.text)

        # Configure Gemini
        # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # model = genai.GenerativeModel(model)

        prompt = f"""Please provide a concise summary of the following code changes. 
        Focus on:
        1. Key functional changes
        2. New features or improvements
        3. Important refactoring
        4. Breaking changes (if any)
        
        Code diff:
        {diff_output}
        """

        response = model.generate_content(prompt)
        print(Markdown(response.text))

    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
