"""Authentication utilities for Identity-Aware Proxy (IAP) access.

This module handles OAuth2 authentication flow with Google's Identity-Aware Proxy (IAP).
It manages the process of obtaining and refreshing IAP tokens by:

1. Prompting users to authenticate via browser-based Google login
2. Handling the OAuth2 callback to receive authorization codes
3. Exchanging codes for access tokens
4. Storing credentials for future use with MLFlow and other IAP-protected services


Typical usage:
    token = get_iap_token()
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from urllib.parse import parse_qs, urlparse

import click
import requests
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token, service_account
from google.oauth2.credentials import Credentials
from rich.console import Console

console = Console()

# OAuth 2.0 client configuration
CLIENT_ID = "938607797672-i7md7k1u3kv89e02b6d8ouo0mi9tscos.apps.googleusercontent.com"
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
SCOPE = ["openid", "email"]
LOCAL_PATH = "conf/local/"

# FastAPI app for callback handling
result_queue = Queue()


def get_service_account_creds(service_account_info: dict) -> service_account.IDTokenCredentials:
    """Get the IAP token credentials for service accounts."""

    token_data = service_account.IDTokenCredentials.from_service_account_info(
        service_account_info, target_audience=CLIENT_ID
    )
    token_data.refresh(google_requests.Request())

    return token_data


def get_user_account_creds() -> Credentials:
    """Gets the IAP token credentials for user accounts, either by using existing credentials or by requesting a new one through a browser OAuth flow."""

    # Try loading existing credentials
    if os.path.exists(f"{LOCAL_PATH}/oauth_token.json"):
        token_data = Credentials.from_authorized_user_file(f"{LOCAL_PATH}/oauth_token.json")
        token_data.refresh(google_requests.Request())

    # if not present, start oauth flow
    else:
        token_data = request_new_iap_token()
        os.makedirs(LOCAL_PATH, exist_ok=True)
        with open(f"{LOCAL_PATH}/oauth_token.json", "w") as f:
            f.write(token_data.to_json())

    return token_data


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth2 callback requests."""

    def do_GET(self):
        """Handle GET requests to the callback endpoint."""
        # Parse query parameters
        query = urlparse(self.path).query
        params = parse_qs(query)

        # Convert from lists to single values
        params = {k: v[0] for k, v in params.items()}

        console.print("Received callback. Putting in queue...")
        result_queue.put(params)

        # Send response
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        response = """
        <html>
        <body>
            <h1>Authentication Complete</h1>
            <p>This window will close automatically. If it doesn't, you can close it manually.</p>
        </body>
        </html>
        """
        self.wfile.write(response.encode())


def start_callback_server(local_port: int) -> int:
    """Start temporary callback server and wait for first request.

    Returns:
        dict: The OAuth callback parameters
    """

    def _run_callback_server():
        server = HTTPServer(("", local_port), OAuthCallbackHandler)
        server.handle_request()  # Handle only one request
        server.server_close()

    server_thread = threading.Thread(target=_run_callback_server)
    server_thread.start()
    return local_port


def get_oauth_client_secret() -> str:
    """Get OAuth client secret from local configuration.

    Returns:
        str: The OAuth client secret
    """
    path = f"{LOCAL_PATH}/oauth_client_secret.txt"

    with open(path) as f:
        return f.read().strip()


def perform_oauth_flow(local_port: int) -> dict:
    """Construct the OAuth URL for user authentication.

    Args:
        port (int): The local server port for callback

    Returns:
        str: The constructed OAuth URL
    """
    successful_port = start_callback_server(local_port)
    auth_url = (
        f"{AUTH_URI}?"
        f"client_id={CLIENT_ID}&"
        f"redirect_uri=http://localhost:{successful_port}/hook&"
        f"scope={' '.join(SCOPE)}&"
        f"response_type=code&"
        f"access_type=offline"
    )
    console.print("Opening browser for authentication...")
    console.print("if the browser does not open, please open it manually and navigate to the following URL:")
    console.print(auth_url)
    click.launch(auth_url)

    # Wait for callback
    result = result_queue.get()
    if "code" not in result:
        raise ValueError("Did not receive authorization code in response")
    return result, successful_port


def exchange_auth_code(auth_code: str, port: int) -> dict:
    """Exchange authorization code for tokens.

    Args:
        auth_code (str): The authorization code from OAuth callback
        port (int): The local server port used for callback
    Returns:
        dict: The token response data

    Raises:
        ValueError: If ID token is not present in response
    """
    CLIENT_SECRET = get_oauth_client_secret()
    token_request = {
        "code": auth_code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": f"http://localhost:{port}/hook",
        "scope": " ".join(SCOPE),
        "grant_type": "authorization_code",
    }

    console.print("Requesting token for user...")
    token_response = requests.post(TOKEN_URI, data=token_request)
    token_data = token_response.json()

    if "id_token" not in token_data:
        raise ValueError("Did not receive ID token in token response")

    return token_data


def _verify_token(id_token_str: str) -> dict:
    """Verify the ID token and get user info.

    Args:
        id_token_str (str): The ID token to verify

    Returns:
        dict: The verified token information
    """
    return id_token.verify_oauth2_token(id_token_str, google_requests.Request(), CLIENT_ID)


def request_new_iap_token(local_port: int = 33333) -> Credentials:
    """Get an Identity-Aware Proxy token through OAuth2 authentication flow.

    See documentation from Google at
    https://cloud.google.com/iap/docs/authentication-howto#authenticating_from_a_desktop_app

    Returns:
        dict: The token data including ID token for authenticating requests

    Raises:
        ValueError: If required tokens are not received in the authentication flow
    """

    CLIENT_SECRET = get_oauth_client_secret()

    # Construct and launch OAuth URL
    code_response, port = perform_oauth_flow(local_port)

    # Exchange code for tokens
    token_data = exchange_auth_code(code_response["code"], port)

    # Verify token and log success
    id_info = _verify_token(token_data["id_token"])
    console.print(f"Successfully authenticated as: {id_info['email']}")

    return Credentials(
        token=token_data["access_token"],
        refresh_token=token_data["refresh_token"],
        id_token=token_data["id_token"],
        token_uri=TOKEN_URI,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPE,
    )
