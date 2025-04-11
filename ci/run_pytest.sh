#!/bin/bash
# Bamboo CI script to install torax-m3 and run all tests
# Note: this script should be run from the root of the git repository

# Debuggging:
set -e -o pipefail
echo "Loading modules:" $@

# Set up environment such that module files can be loaded
. /usr/share/Modules/init/sh
module purge
# Modules are supplied as arguments in the CI job:
module load $@

# Debuggging:
echo "Done loading modules"
set -x


# Set up the testing venv
rm -rf venv  # Environment should be clean, but remove directory to be sure
python -m venv venv
source venv/bin/activate

# Install torax-m3 and test dependencies
pip install --upgrade pip setuptools wheel
pip install .[test]

# Debugging:
pip freeze

# Run pytest
# Clean artifacts created by pytest
rm -f junit.xml
rm -rf htmlcov
export IMAS_VERSION="4.0.0" # set default version because IMAS-Python module on SDCC doesn't have latest one yet
python -m pytest -n=auto --cov=torax-m3 --cov-report=term-missing --cov-report=html --junit-xml=junit.xml
