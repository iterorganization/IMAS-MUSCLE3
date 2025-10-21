.. _`installing`:

Installing IMAS-MUSCLE3
=======================

SDCC installation
-------------------

* Setup a project folder and clone git repository

  .. code-block:: bash

    mkdir projects
    cd projects
    git clone git@github.com:iterorganization/IMAS-MUSCLE3.git
    cd IMAS-MUSCLE3

* Setup a python virtual environment and install python dependencies

  .. code-block:: bash

    module load IMAS-AL-Core

    python3 -m venv ./venv
    . venv/bin/activate
    pip install --upgrade pip
    pip install --upgrade wheel setuptools
    # For development an installation in editable mode may be more convenient
    pip install .[all]

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

    # Make sure IMAS-Core is installed and available
    # for local IMAS-Core installation
    git clone ssh://git@git.iter.org/imas/al-core.git -b main
    pip install ./al-core

    python3 -c "import imas_muscle3; print(imas_muscle3.__version__)"
    pytest

Documentation
-------------

* To build the IMAS-MUSCLE3 documentation, execute:

  .. code-block:: bash

    make -C docs html
