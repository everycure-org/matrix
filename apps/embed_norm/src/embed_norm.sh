#!/bin/bash

cd /home/wadmin/embed_norm/apps/embed_norm/src/

source "$(find "$(git rev-parse --show-toplevel)" -type f -name 'activate' -path '*/bin/activate' | head -n 1)"

exec python main.py