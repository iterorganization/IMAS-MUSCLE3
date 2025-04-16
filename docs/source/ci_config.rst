.. _`ci configuration`:

CI configuration
================

IMAS-MUSCLE3 uses github for CI. This page provides an overview
of the CI Plan and deployment projects.

CI Plan
-------

The `IMAS-MUSCLE3 CI plan <https://ci.iter.org/browse/IC-PYM>`_ consists of 3 types of jobs:

Linting 
    Run ``black``, ``flake8``, ``mypy`` and ``isort`` on the PDS code base.
    See :ref:`code style and linting`.

Testing
    This runs all unit tests with pytest.

Build docs
    This job builds the Sphinx documentation.
