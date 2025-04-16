.. _`installing`:

Installing IMAS-MUSCLE3
=================================

SDCC setup
----------

* Quick developer installation guide

  For ITER SDCC

  .. code-block:: bash

    # Load modules every time you use IMAS-MUSCLE3
    module load IMAS-Python MUSCLE3
    python3 -m venv ./venv
    . venv/bin/activate
    pip install -e .[all]
    python3 -c "import imas_muscle3; print(imas_muscle3.__version__)"
    pytest

Ubuntu installation
-------------------

* Install system packages

  .. code-block:: bash

    sudo apt update
    sudo apt install build-essential git-all python3-dev python-is-python3 \
      python3 python3-venv python3-pip python3-setuptools

* Setup a project folder and clone git repository

  .. code-block:: bash

    mkdir projects
    cd projects
    git clone git@github.com:iterorganization/IMAS-MUSCLE3.git
    cd IMAS-MUSCLE3

* Setup a python virtual environment and install python dependencies

  .. code-block:: bash

    python3 -m venv ./venv
    . venv/bin/activate
    pip install --upgrade pip
    pip install --upgrade wheel setuptools
    # For development an installation in editable mode may be more convenient
    pip install .[all]
    python3 -c "import torax-m3; print(torax-m3.__version__)"
    pytest

Documentation
-------------

* To build the IMAS-MUSCLE3 documentation, execute:

  .. code-block:: bash

    make -C docs html
