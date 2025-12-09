.. _`release_imas`:

IMAS-MUSCLE3 development and release process
============================================

IMAS-MUSCLE3 development follows the fork-based model described in
`the contributing guidelines
<https://github.com/iterorganization/IMAS-MUSCLE3/blob/develop/CONTRIBUTING.md>`_.


Creating an IMAS-MUSCLE3 release
--------------------------------

1.  Create a Pull Request from ``develop`` to ``main``.
2.  Add a change log to the Pull Request, briefly describing new features, bug fixes,
    and update accordingly the :ref:`changelog`.
3.  The PR is reviewed and merged by the maintainers who also create the release tags.
4.  After the release PR is merged, update the Easybuild configurations for SDCC modules
    in the `easybuild-easyconfigs repository
    <https://github.com/easybuilders/easybuild-easyconfigs>`_.
    See :ref:`packaging` for more details on how to do this.
