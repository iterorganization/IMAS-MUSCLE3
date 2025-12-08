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
    See the next section for more details on how to do this.


Updating and testing the IMAS-MUSCLE3 Easybuild configuration
-------------------------------------------------------------

The following steps must be performed for each of the supported tool chains
(currently ``intel-2023b``, ``foss-2023b``):

1.  Create the ``.eb`` file for the new release.

    a.  Copy the ``.eb`` file from the previous release.
    b.  Update the ``version`` to reflect the just-released version tag.
    c.  If any of the IMAS-MUSCLE3 dependencies in ``pyproject.toml`` where updated or changed
        since the previous release, update the easybuild dependencies:

        -   ``builddependencies`` contains build-time dependencies which are available
            as a module on SDCC.
        -   ``dependencies`` contains run-time dependencies which are available as a
            module on SDCC.
        -   ``exts_list`` contains python package dependencies (and potentially
            dependencies of dependencies) which are not available in any of the Python
            modules on SDCC.
    
    d.  Update the checksum of imas: download an archive of the IMAS-MUSCLE3 repository from
        bitbucket. This is easiest to do by copying the following URL, replace
        ``<version>`` with the version tag, and paste it in a web browser:

        .. code-block:: text

            https://github.com/iterorganization/IMAS-MUSCLE3/archive/refs/tags/<version>.tar.gz

        Then, calculate the hash of the downloaded archive with ``sha256sum`` and update
        it in the ``.eb`` file.

2.  Test the easybuild configuration:

    a.  Create an easybuild module, replace ``<eb_file>`` with the filename of the
        ``.eb`` file created in step 1.

        .. code-block:: bash

            module purge
            module load EasyBuild
            eb --rebuild <eb_file>

        If this is unsuccessful, investigate the error and update the ``.eb``
        configuration. A useful environment variable for debugging is ``export
        PIP_LOG=pip.log``, which instructs pip to write logs to the specified file
        (``pip.log`` in this example).
    b.  If the module was successfully installed by easybuild, load it:

        .. code-block:: bash

            module purge
            module use ~/.local/easybuild/modules/all/
            module load IMAS-MUSCLE3/<version>-<toolchain>
            module load IMAS-AL-Core
    
    c.  Sanity check the module, for example by running the ``pytest`` unit tests.

