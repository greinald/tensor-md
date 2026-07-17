#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./publish.sh testpypi
#   ./publish.sh pypi
# Credentials are read by Twine from its normal prompt, environment, or
# ~/.pypirc. No API key is stored in this repository.

repository="${1:-pypi}"
rm -rf dist build tensor_md.egg-info
python -m build
python -m twine check dist/*

if [[ "$repository" == "testpypi" ]]; then
    python -m twine upload --repository testpypi dist/*
elif [[ "$repository" == "pypi" ]]; then
    python -m twine upload dist/*
else
    echo "Usage: $0 [testpypi|pypi]" >&2
    exit 2
fi
