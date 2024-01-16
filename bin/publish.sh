#!/bin/bash

# Exit in case of error
set -e

# Optional: Activate your virtualenv
source venv-dev/bin/activate

# Check if twine is installed
if ! [ -x "$(command -v twine)" ]; then
  echo 'Error: twine is not installed.' >&2
  exit 1
fi

# Clean up the build artifacts
rm -rf dist    

# Bump the version, build and publish to PyPI
# You can use bump2version or manually update the version in your setup.py or pyproject.toml

# Build the package
python setup.py sdist bdist_wheel

# Publish the package to PyPI
twine upload dist/*

# Clean up the build artifacts
rm -rf build dist *.egg-info
