#!/usr/bin/python
# requires typer, tqdm, requests
import subprocess
import sys

import typer
from tqdm import tqdm

app = typer.Typer(help="GitHub user management commands", no_args_is_help=True)

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


@app.command(name="bulk-add", help="Add multiple users to a single team")
def add_users_to_team(
    org: str = typer.Option(help="The name of the GitHub organization.", default="everycure-org"),
    team: str = typer.Argument(..., help="The slug of the team in the organization."),
):
    """Adds users to an organization team from a CSV input list."""
    print(
        """
please enter a list of users newline separated and hit ctrl-d to drop the names
-------------------------------------------------------------------------------"""
    )
    users = [line.strip() for line in sys.stdin if line.strip()]

    with tqdm(total=len(users), desc="Adding users to team") as pbar:
        for user in users:
            add_user_to_team(org, team, user)
            pbar.update(1)


@app.command(name="add", help="Add a user to multiple teams")
def add_user_to_teams(
    user: str = typer.Argument(..., help="The GitHub username to add to teams."),
    teams: str = typer.Argument(..., help="Comma-separated list of team slugs to add the user to."),
    org: str = typer.Option(help="The name of the GitHub organization.", default="everycure-org"),
):
    """Adds a single user to multiple organization teams.

    Args:
        user: The GitHub username to add.
        teams: Comma-separated list of team slugs.
        org: The GitHub organization name.

    Example:
        matrix gh-users add johndoe "team1,team2,team3"
    """
    team_list = [team.strip() for team in teams.split(",") if team.strip()]

    if not team_list:
        typer.echo("Error: No valid teams provided", err=True)
        raise typer.Exit(1)

    with tqdm(total=len(team_list), desc=f"Adding user {user} to teams") as pbar:
        for team in team_list:
            add_user_to_team(org, team, user)
            pbar.update(1)


if __name__ == "__main__":
    app()
