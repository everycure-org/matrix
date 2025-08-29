#!/bin/bash
set -e
curl -fsSL https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.11 if not available
if ! pyenv versions | grep -q "3.11"; then
pyenv install 3.11
fi
pyenv global 3.11

curl -LsSf https://astral.sh/uv/install.sh | sh