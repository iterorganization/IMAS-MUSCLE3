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

    The CI script executed in this job is ``ci/linting.sh``.

Testing
    This runs all unit tests with pytest.

    The CI script executed in this job is ``ci/run_pytest.sh``, which expects the
    modules it needs to load as arguments.

Build docs
    This job builds the Sphinx documentation.

    The CI script executed in this job is: ``ci/build_docs_and_dist.sh``, which expects the
    modules it needs to load as arguments.
