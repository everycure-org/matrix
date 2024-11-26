import re
from pathlib import Path
from subprocess import CalledProcessError

import typer

from matrix_cli.components.settings import settings
from matrix_cli.components.utils import console, get_git_root, run_command

secrets = typer.Typer(help="A set of utility commands for managing secrets", no_args_is_help=True)


@secrets.command(name="rotate-pk")
def rotate_pk(user_email: str, key_name: str = typer.Argument(default="default")):
    # this function rotates out the PK used by the repository
    # this a potentially destructive operation so we act with prudence here and prompt the user for confirmation before executing
    # However we leverage an existing community shell script here so we don't have to implement the logic ourselves
    user_key_id = get_gpg_key_id(user_email)

    # check we're not on main, this should always happen on a branch
    if run_command(["git", "branch", "--show-current"], log_before=True) == "main":
        console.print("[bold red] You are on the main branch, please create a branch first")
        exit(1)

    # wipe the key for the user to remove
    console.print("Making sure the user will not be added again by wiping their key")
    wipe_key_for_user_to_remove(key_name, user_key_id, user_email)
    # any public key we have gets imported first
    console.print("Ensuring any public key is imported")
    import_all_keys_into_keychain()
    # if we are still missing people here, we need to abort
    console.print("Ensuring all remaining individuals retain access")
    ensure_all_keys_imported(key_name)

    console.print(f"[bold green] Rotating out GPG, removing access for {user_email}")
    console.print("[bold red] This is a fairly destructive operation, please ensure you have backups")
    if not typer.confirm(
        "A number of changes have been made to the repository, please confirm these by committing and then continue"
    ):
        console.print("[bold red] Aborting")
        exit(1)

    # check no changes are staged
    # Check for any uncommitted changes (both staged and unstaged)
    if run_command(["git", "status", "--porcelain"]):
        console.print("[bold red] There are uncommitted changes, please commit them first")
        exit(1)
    perform_actual_rotation(user_key_id, key_name)


def perform_actual_rotation(user_key_id: str, key_name: str):
    remove_user_script = Path(get_git_root()) / "apps" / "matrix-cli" / "tools/remove-gpg-user.sh"
    try:
        run_command([remove_user_script, user_key_id, key_name], log_before=True)
    except CalledProcessError as ex:
        console.print(ex.stdout)
        console.print(ex.stderr)
        console.print("[bold red] Failed to remove user, exiting")
        exit(1)


@secrets.command(name="list")
def list(key_name: str = typer.Argument(default="default")):
    """Lists the names/Emails of the users that have access to the given key"""
    # first we ensure all keys are imported
    import_all_keys_into_keychain()
    # then we list the users
    list_user_ids_with_access(key_name)


@secrets.command(name="sync-pks")
def sync_pks(key_name: str = typer.Argument(help="The key to sync the users for")):
    """Synchronizes the public keys of the users with the keys in the repository.

    This command will iterate over all users that have access to the given key and ensure their public keys are present in the repository.
    """
    user_ids = get_user_ids_for_key(key_name)

    # usernames are somename <email@domain.com>, we care only about the email
    email_regex = r"<(.+@.+)>"

    for user_id in user_ids:
        # get the email for the user
        username_string = get_username_for_id(user_id)
        try:
            email = re.search(email_regex, username_string).group(1)
        except Exception:
            console.print(f"[bold red] Could not find email for {user_id}")
            continue
        filename = Path(get_git_root()) / settings.gpg_public_key_path / f"{email}.asc"
        # export the key of the user to a .asc file in the target_key directory
        console.print(f"[bold green] Exporting key for {email} to {filename}")
        run_command(["gpg", "--export", "--armor", "--output", str(filename), email], log_before=True)


@secrets.command(name="import-key")
def import_key(file_path: str = typer.Argument(help="The path to the key to import")):
    # imports a key into the keychain
    try:
        path = Path(file_path).expanduser()
        stdout, stderr = run_command(f"gpg --import '{path}'", include_stderr=True)
        console.print(stdout)
        console.print(stderr)
    except CalledProcessError as ex:
        console.print(ex.stdout)
        console.print(ex.stderr)
        console.print("[bold red] Failed to import key")
        exit(1)

    console.print("[bold green] Key imported")
    console.print("[bold green] Ensuring ownertrust")

    # getting all fingerprints from file
    stdout = run_command(f"gpg --with-colons --import-options show-only --import '{path}'")
    fingerprints = [row.split(":")[9] for row in stdout.split("\n") if "fpr" in row]

    for fingerprint in fingerprints:
        console.print(f"[bold green] Ensuring ownertrust for {fingerprint}")
        run_command(f"echo {fingerprint}:6: | gpg --import-ownertrust")


def wipe_key_for_user_to_remove(key_name: str, user_key_id: str, user_email: str):
    path = Path(get_git_root()) / settings.gpg_key_path / key_name / "0" / f"{user_key_id}.gpg"
    console.print(f"[bold green] Wiping key for {user_key_id} in path {path}")
    if path.exists():
        run_command(f"rm -rf {path}", check=False)
    else:
        console.print(f"[bold yellow] Key not found for {user_key_id}, skipping")

    # next deleting the users .asc file
    filename = Path(get_git_root()) / settings.gpg_public_key_path / f"{user_email}.asc"
    if filename.exists():
        run_command(f"rm -rf {filename}", check=False)
    else:
        if not typer.confirm(
            "The public key for the user to remove was not found, please remove it manually, Then hit y to continue"
        ):
            console.print("User did not confirm, aborting")
            exit(1)


def ensure_all_keys_imported(key_name: str):
    # ensures all keys present in the target_key directory are in our gpg keychain
    path = str(Path(get_git_root()) / settings.gpg_key_path / key_name / "0")
    keys = run_command(f"ls {path}", check=False, log_before=True)
    key_ids = [key.split(".")[0] for key in keys.split("\n")]  # crop away .gpg

    keys_missing = []
    for key in key_ids:
        console.print(f"[bold green] Checking if {key} is in the keychain")
        # this crashes if the key is not in the keychain
        try:
            run_command(["gpg", "--list-keys", key])
        except Exception:
            keys_missing.append(key)

    if len(keys_missing) > 0:
        console.print(f"[bold red] {len(keys_missing)} not found in keychain, make sure you have imported them")
        console.print("\n".join([f"- {k}" for k in keys_missing]))
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
        run_command([path, abs_file_path])
        console.print(f"Ensured {file} is in the keychain with full trust")

    console.print("[bold green] All keys imported")


def list_user_ids_with_access(key_name: str):
    # lists the user IDs with access to the given key
    files = get_user_ids_for_key(key_name)

    # Call gpg --list-keys for each file
    for user_id in files:
        try:
            username = get_username_for_id(user_id)
            if username is None:
                console.print(f"[bold red]: Could not find UID for {user_id}")
            else:
                console.print(f"- {username}")
        except Exception:
            console.print(f"[bold red]Warning: Could not find key for {user_id}")


def get_user_ids_for_key(key_name):
    base_path = Path(get_git_root()) / settings.gpg_key_path / key_name / "0"
    console.print(f"[bold green] Listing user IDs with access to {key_name} in path {base_path}")
    # Get list of files in directory
    files = [f.stem for f in base_path.glob("*")]
    return files


def get_username_for_id(user_id: str):
    result = run_command(["gpg", "--list-keys", "--with-colons", user_id], check=False)
    uid_row = next((row for row in result.split("\n") if "uid" in row), None)
    if uid_row is None:
        return None
    else:
        return uid_row.split(":")[9]
