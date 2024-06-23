#!/usr/bin/python
# requires typer, tqdm, requests
import typer
from pathlib import Path
from tqdm import tqdm
import subprocess

app = typer.Typer()

GITHUB_API_URL = "https://api.github.com"


def add_user_to_team(org: str, team: str, user: str):
    """Adds a user to a team in a GitHub organization using the gh CLI."""
    try:
        subprocess.check_output(
            [
                "gh",
                "api",
                "--method",
                "PUT",
                f"/orgs/{org}/teams/{team}/memberships/{user}",
                "-f",
                "role=member",
            ],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error adding user {user} to team {team}: {e.stderr.decode()}")


@app.command()
def add_users_to_team(
    users_file: Path = typer.Argument(
        ..., help="Path to the CSV file containing the list of users."
    ),
    org: str = typer.Option(
        help="The name of the GitHub organization.", default="everycure-org"
    ),
    team: str = typer.Argument(..., help="The slug of the team in the organization."),
):
    """Adds users to an organization team from a CSV input list."""

    if not users_file.exists():
        typer.echo(f"Error: File {users_file} does not exist.")
        raise typer.Exit(code=1)

    with open(users_file, "r") as f:
        users = [line.strip() for line in f if line.strip()]

    with tqdm(total=len(users), desc="Adding users to team") as pbar:
        for user in users:
            add_user_to_team(org, team, user)
            pbar.update(1)


if __name__ == "__main__":
    app()
