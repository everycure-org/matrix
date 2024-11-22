from pathlib import Path

import typer

from matrix_cli.components.settings import settings
from matrix_cli.components.utils import console, get_git_root, run_command

secrets = typer.Typer(help="A set of utility commands for managing secrets", no_args_is_help=True)


@secrets.command(name="rotate-pk")
def rotate_pk(user_email: str, key_name: str = typer.Argument(default="default")):
    # this function rotates out the PK used by the repository
    # this a potentially destructive operation so we act with prudence here and prompt the user for confirmation before executing
    # However we leverage an existing community shell script here so we don't have to implement the logic ourselves
    path = Path(get_git_root()) / "apps" / "matrix-cli" / "tools/remove-gpg-user.sh"
    user_key_id = get_gpg_key_id(user_email)

    # check we're not on main, this should always happen on a branch
    if run_command(["git", "branch", "--show-current"], log_before=True) == "main":
        console.print("[bold red] You are on the main branch, please create a branch first")
        exit(1)

    # wipe the key for the user to remove
    console.print("Making sure the user will not be added again by wiping their key")
    wipe_key_for_user_to_remove(key_name, user_key_id)
    # any public key we have gets imported first
    console.print("Ensuring any public key is imported")
    import_all_keys_into_keychain()
    # if we are still missing people here, we need to abort
    console.print("Ensuring all remaining individuals retain access")
    ensure_all_keys_imported(key_name)

    console.print(f"[bold green] Rotating out GPG, removing access for {user_email}")
    console.print("[bold red] This is a fairly destructive operation, please ensure you have backups")
    if not typer.confirm("Are you sure you want to continue?"):
        console.print("[bold red] Aborting")
        exit(1)

    run_command(["bash", "-c", path, user_key_id, key_name], log_before=True)


def wipe_key_for_user_to_remove(key_name: str, user_key_id: str):
    path = Path(get_git_root()) / settings.gpg_key_path / key_name / "0" / f"{user_key_id}.gpg"
    if path.exists():
        run_command(f"rm -rf {path}")
    else:
        console.print(f"[bold red] Key not found for {user_key_id}, skipping")
        exit(1)


def ensure_all_keys_imported(key_name: str):
    # ensures all keys present in the target_key directory are in our gpg keychain
    keys = run_command(f"ls {get_git_root() / settings.gpg_key_path / key_name / '0'}")
    key_ids = [key.split(".")[0] for key in keys.split("\n")]  # crop away .gpg

    for key in key_ids:
        console.print(f"[bold green] Checking if {key} is in the keychain")
        # this crashes if the key is not in the keychain
        try:
            run_command(["gpg", "--list-keys", key])
        except Exception:
            console.print(f"[bold red] Key {key} not found in keychain, make sure you have imported it")
            console.print("you have imported it first. We cannot proceed without this")
            exit(1)


def get_gpg_key_id(user_email: str):
    # gets the GPG key ID for a given user email
    console.print(f"[bold green] Getting GPG key ID for {user_email}")
    command = ["gpg", "--list-keys", "--with-colons", user_email]
    result = run_command(command, log_before=True)
    rows = result.split("\n")
    key_count = len([x for x in rows if "tru::" in x])
    if key_count > 1:
        console.print("We can only deal with 1 key per user, if you have more keys, please delete any stale ones")
        console.print(f"User in question: {user_email}")
        console.print(f"Number of keys present: {key_count}")
        exit(1)

    # extract the key ID
    key = next(x for x in rows if "fpr" in x).split(":")[9]
    return key

    key = run_command(["gpg", "--list-keys", user_email])
    console.print(f"[bold green] GPG key ID: {key}")
    return key


def import_all_keys_into_keychain():
    # imports all keys
    base_path = Path(get_git_root()) / settings.gpg_public_key_path

    # iterate over all files in list
    for file in base_path.glob("**/*.asc"):
        # import each into gpg and add add ultimate trust
        abs_file_path = file.absolute()
        path = Path(get_git_root()) / "apps" / "matrix-cli" / "tools" / "import-pub-key-full-trust.sh"
        run_command(["bash", "-c", f"{path} {abs_file_path}"])
        console.print(f"Ensured {file} is in the keychain with full trust")

    console.print("[bold green] All keys imported")
