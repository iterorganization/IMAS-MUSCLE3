.. _`ci configuration`:

CI configuration
================

IMAS-MUSCLE3 uses github for CI. This page provides an overview
of the CI Plan and deployment projects.

CI Plan
-------

The IMAS-MUSCLE3 CI plan consists of 3 types of jobs:

Linting 
    Run ``python -m ty check imas_muscle3``, ``ruff check --fix`` and ``ruff format`` on the IMAS-MUSCLE3 code base.
    See :ref:`code style and linting`.

Testing
    This runs all unit tests with pytest.

Build docs
    This job builds the Sphinx documentation.
