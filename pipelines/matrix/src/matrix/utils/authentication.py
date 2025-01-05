import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from urllib.parse import parse_qs, urlparse

import click
import requests
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from rich.console import Console

console = Console()

# OAuth 2.0 client configuration
CLIENT_ID = "938607797672-i7md7k1u3kv89e02b6d8ouo0mi9tscos.apps.googleusercontent.com"
CLIENT_SECRET = "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"
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
LOCAL_PATH = "data/.credentials/"
SCOPE = ["openid", "email"]

# FastAPI app for callback handling
result_queue = Queue()


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


def start_callback_server() -> int:
    """Start temporary callback server and wait for first request.

    Returns:
        dict: The OAuth callback parameters
    """

    def _run_callback_server(port):
        server = HTTPServer(("", port), OAuthCallbackHandler)
        server.handle_request()  # Handle only one request

    for i in range(4000, 4010):
        # try starting the server on up to 10 ports before giving up
        try:
            server_thread = threading.Thread(target=_run_callback_server, args=(i,))
            # server_thread.daemon = True
            server_thread.start()
            print(f"Started server on port {i}")
            return i
        except Exception as e:
            console.print(f"Failed to start server on port {i}: {e}")

        raise Exception("Failed to start callback server")


def get_iap_token() -> str:
    """Gets the IAP token, either by using existing credentials or by requesting a new one through a browser OAuth flow."""
    return request_new_iap_token()


def request_new_iap_token() -> str:
    """Get an Identity-Aware Proxy token through OAuth2 authentication flow.

    See documentation from Google at
    https://cloud.google.com/iap/docs/authentication-howto#authenticating_from_a_desktop_app


    This function will:
    1. Open a browser for user authentication
    2. Start a local server to receive the callback
    3. Exchange the auth code for tokens
    4. Return the ID token for use with IAP-protected resources

    Returns:
        str: The ID token for authenticating requests

    Raises:
        ValueError: If required tokens are not received in the authentication flow
    """

    # start the local server
    port = start_callback_server()
    # Start authentication flow
    # Construct OAuth URL
    auth_url = (
        f"{AUTH_URI}?"
        f"client_id={CLIENT_ID}&"
        f"redirect_uri=http://localhost:{port}/hook&"
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

    # Exchange authorization code for tokens
    token_request = {
        "code": result["code"],
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

    # Verify the ID token
    id_info = id_token.verify_oauth2_token(token_data["id_token"], google_requests.Request(), CLIENT_ID)
    console.print(f"Successfully authenticated as: {id_info['email']}")

    return token_data["id_token"]
