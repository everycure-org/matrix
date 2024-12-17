import typer

from matrix_cli.commands.code import app as code_app
from matrix_cli.commands.data import data_app
from matrix_cli.commands.gh_users import app as gh_users_app
from matrix_cli.commands.releases import app as releases_app

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

app.add_typer(
    data_app,
    name="data",
    help="Data-related utility commands",
)

if __name__ == "__main__":
    app()
