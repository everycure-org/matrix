import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import core_entities.resolvers as resolvers
from core_entities.resolvers import purple_book_url

_BASE_URL = resolvers._PURPLE_BOOK_BASE_URL


def test_purple_book_url_defaults_to_previous_utc_month_and_year(monkeypatch) -> None:
    now_utc = datetime(2026, 3, 25, tzinfo=timezone.utc)
    monkeypatch.setattr(resolvers, "_resolve_now_utc", lambda: now_utc)
    monkeypatch.setattr(resolvers, "_is_url_available", lambda url, timeout_seconds=5.0: True)

    last_month_utc = now_utc.replace(day=1) - timedelta(days=1)
    expected_month = last_month_utc.strftime("%B")
    expected_year = str(last_month_utc.year)

    result = purple_book_url()

    assert f"/{expected_year}/" in result
    assert f"purplebook-search-{expected_month}-data-download.csv" in result


def test_purple_book_url_falls_back_when_last_month_is_unavailable(monkeypatch) -> None:
    fixed_now = datetime(2026, 3, 25, tzinfo=timezone.utc)
    monkeypatch.setattr(resolvers, "_resolve_now_utc", lambda: fixed_now)

    checked_urls = []

    def fake_is_url_available(url: str, timeout_seconds: float = 5.0) -> bool:
        del timeout_seconds
        checked_urls.append(url)
        if "-February-" in url:
            return False
        if "-January-" in url:
            return True
        return False

    monkeypatch.setattr(resolvers, "_is_url_available", fake_is_url_available)

    result = purple_book_url()

    assert result == f"{_BASE_URL}/2026/purplebook-search-January-data-download.csv"
    assert checked_urls[0].endswith("/2026/purplebook-search-February-data-download.csv")
    assert checked_urls[1].endswith("/2026/purplebook-search-January-data-download.csv")


def test_purple_book_url_fallback_adjusts_year_across_january_boundary(monkeypatch) -> None:
    fixed_now = datetime(2026, 1, 25, tzinfo=timezone.utc)
    monkeypatch.setattr(resolvers, "_resolve_now_utc", lambda: fixed_now)

    def fake_is_url_available(url: str, timeout_seconds: float = 5.0) -> bool:
        del timeout_seconds
        return "-November-" in url

    monkeypatch.setattr(resolvers, "_is_url_available", fake_is_url_available)

    result = purple_book_url()

    assert result == f"{_BASE_URL}/2025/purplebook-search-November-data-download.csv"


def test_purple_book_url_uses_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("FDA_PURPLE_BOOK_YEAR", "2024")
    monkeypatch.setenv("FDA_PURPLE_BOOK_MONTH", "JANUARY")

    result = purple_book_url()

    assert result == f"{_BASE_URL}/2024/purplebook-search-January-data-download.csv"


def test_purple_book_url_prefers_explicit_arguments_over_environment(monkeypatch) -> None:
    monkeypatch.setenv("FDA_PURPLE_BOOK_YEAR", "2024")
    monkeypatch.setenv("FDA_PURPLE_BOOK_MONTH", "january")

    result = purple_book_url(year="2025", month="March")

    assert result == f"{_BASE_URL}/2025/purplebook-search-March-data-download.csv"
