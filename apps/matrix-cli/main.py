#!/usr/bin/env python3
# NOTE: This file was partially generated using AI assistance.

import typer
from modules.gh_users import app as gh_users_app
from modules.releases import app as releases_app
from modules.code import app as code_app

app = typer.Typer(
    help="CLI tools for managing GitHub and releases",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(
    gh_users_app,
    name="gh-users",
    help="Manage GitHub users and teams",
)

app.add_typer(
    releases_app,
    name="releases",
    help="Manage releases and release notes",
)

app.add_typer(
    code_app,
    name="code",
    help="Code-related utility commands",
)

if __name__ == "__main__":
    app()
