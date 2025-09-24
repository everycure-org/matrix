#!/usr/bin/env python3
"""
Update version and dateModified (YYYY-MM-DD) in codemeta.json
based on latest git tag and commit date.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

CODEMETA = Path("codemeta.json")

def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def get_latest_tag() -> str:
    tag = run(["git", "describe", "--tags", "--abbrev=0"])
    return tag.lstrip("v") if tag else "0.0.0"

def get_last_commit_date() -> str:
    ts = run(["git", "log", "-1", "--format=%ci"])
    if ts:
        # %ci → "YYYY-MM-DD HH:MM:SS +ZZZZ"
        return ts.split()[0]
    return datetime.utcnow().strftime("%Y-%m-%d")

def main():
    if not CODEMETA.exists():
        raise FileNotFoundError("codemeta.json not found")

    data = json.loads(CODEMETA.read_text(encoding="utf-8"))
    data["version"] = get_latest_tag()
    data["dateModified"] = get_last_commit_date()

    CODEMETA.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated codemeta.json → version={data['version']}, dateModified={data['dateModified']}")

if __name__ == "__main__":
    main()
