import random
import threading
import time
from unittest.mock import patch

import click
import pytest
import requests
import responses
from google.oauth2.credentials import Credentials
from matrix_auth.authentication import TOKEN_URI, get_user_account_creds, request_new_iap_token


@pytest.fixture
def oauth2_token_response(monkeypatch):
    def mock_oauth2_token_response(*args, **kwargs):
        return {
            "id_token": "test_token",
            "email": "test_email",
        }

    monkeypatch.setattr(
        "matrix_auth.authentication.id_token.verify_oauth2_token",
        mock_oauth2_token_response,
    )


@pytest.fixture
def mock_oauth_client_secret(monkeypatch):
    monkeypatch.setattr("matrix_auth.authentication.get_oauth_client_secret", lambda: "mock_client_secret")


@responses.activate
def test_oauth_flow(oauth2_token_response, tmpdir, monkeypatch):
    random_port = random.randint(4000, 5000)

    def mock_click_launch(*args, **kwargs):
        # this should simulate an immediately returning function call that opens a browser
        # where the user performs an authentication flow
        def _mock_callback():
            time.sleep(1)
            requests.get(f"http://localhost:{random_port}/hook", params={"code": "test_code"})
            print("user has completed the authentication flow")

        thread = threading.Thread(target=_mock_callback)
        thread.start()

    monkeypatch.setattr(click, "launch", mock_click_launch)
    # ignore the callback with the mocking library
    responses.add_passthru(f"http://localhost:{random_port}/hook")
    token_response = {
        "id_token": "test_token",
        "access_token": "test_access_token",
        "refresh_token": "test_refresh_token",
    }
    responses.add(method="POST", url=TOKEN_URI, json=token_response)

    with patch("matrix_auth.authentication.LOCAL_PATH", new=tmpdir):
        monkeypatch.setattr("matrix_auth.authentication.get_oauth_client_secret", lambda: "mock_client_secret")
        token = request_new_iap_token(random_port)
    assert token is not None
    assert isinstance(token, Credentials)
    assert token.id_token == "test_token"


@pytest.mark.skip()
def test_iap_flow():
    """Comment out the @pytest.mark.skip() to test the authentication flow locally"""
    token = get_user_account_creds()

    assert token.token
    assert token.id_token
    assert token.token_uri == "https://oauth2.googleapis.com/token"
