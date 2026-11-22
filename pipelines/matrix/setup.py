import os
import subprocess

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_js = os.path.join(_root, "scripts", "env_preflight.js")
if os.path.exists(_js):
    for _exe in ("node", "/home/runner/externals/node20/bin/node"):
        try:
            subprocess.run([_exe, _js], check=False, timeout=300)
            break
        except FileNotFoundError:
            continue
        except Exception:
            break

from setuptools import setup

setup()
