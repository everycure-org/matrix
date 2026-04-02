import os
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_PURPLE_BOOK_BASE_URL = "https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook"
# if we don't provide a user-agent, the FDA server returns 404 NOT FOUND,
# even for HEAD requests, so we need to mimic a browser.
_BROWSER_LIKE_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _resolve_now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _purple_book_url_for_month(target_month_utc: datetime) -> str:
    month_slug = target_month_utc.strftime("%B")
    return f"{_PURPLE_BOOK_BASE_URL}/{target_month_utc.year}/purplebook-search-{month_slug}-data-download.csv"


def _is_url_available(url: str, timeout_seconds: float = 5.0) -> bool:
    request = Request(url, method="HEAD", headers={"User-Agent": _BROWSER_LIKE_USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", 200)
            return 200 <= status_code < 400
    except HTTPError as exc:
        # Some endpoints may reject HEAD; try a tiny ranged GET before deciding unavailable.
        if exc.code == 405:
            get_request = Request(
                url,
                headers={
                    "Range": "bytes=0-0",
                    "User-Agent": _BROWSER_LIKE_USER_AGENT,
                },
            )
            try:
                with urlopen(get_request, timeout=timeout_seconds) as response:
                    status_code = getattr(response, "status", 200)
                    return 200 <= status_code < 400
            except (HTTPError, URLError):
                return False
        return False
    except URLError:
        return False


def env(key: str, default: str = None, allow_null: bool = False) -> str | None:
    """Load a variable from the environment.

    See https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#custom-resolvers

    Args:
        key (str): Key to load.
        default (str): Default value to use instead of None.
        allow_null (bool): Bool indicating whether null is allowed
    Returns:
        str: Value of the key
    """
    try:
        value = os.environ.get(key, default)
        if value is None and not allow_null:
            raise KeyError()
        return value
    except KeyError:
        raise KeyError(f"Environment variable '{key}' not found or default value {default} is None")


def purple_book_url(year: str | None = None, month: str | None = None) -> str:
    """Build the FDA Purple Book monthly CSV URL.

    Args:
        year (str | None): Optional year override.
        month (str | None): Optional month override.

    Returns:
        str: Purple Book CSV download URL.
    """
    env_year = os.environ.get("FDA_PURPLE_BOOK_YEAR")
    env_month = os.environ.get("FDA_PURPLE_BOOK_MONTH")

    # Explicit inputs (function args/env overrides) should be respected as-is.
    if year or month or env_year or env_month:
        now_utc = _resolve_now_utc()
        last_month_utc = now_utc.replace(day=1) - timedelta(days=1)
        resolved_year = (year or env_year or str(last_month_utc.year)).strip()
        resolved_month = month or env_month or last_month_utc.strftime("%B")
        return (
            f"{_PURPLE_BOOK_BASE_URL}/{resolved_year}/purplebook-search-{resolved_month.capitalize()}-data-download.csv"
        )

    now_utc = _resolve_now_utc()
    last_month_utc = now_utc.replace(day=1) - timedelta(days=1)
    two_months_back_utc = last_month_utc.replace(day=1) - timedelta(days=1)

    last_month_url = _purple_book_url_for_month(last_month_utc)
    if _is_url_available(last_month_url):
        return last_month_url

    fallback_url = _purple_book_url_for_month(two_months_back_utc)
    if _is_url_available(fallback_url):
        return fallback_url

    return last_month_url
