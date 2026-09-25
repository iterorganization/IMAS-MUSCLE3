.. _`usage`:

Using IMAS-MUSCLE3
==================

IMAS-MUSCLE3 helper actors can be used like any other MUSCLE3 actors.
For a quick reminder on MUSCLE3 workflows, see the `documentation <https://muscle3.readthedocs.io/en/latest/index.html>`_.

Importing the actors
--------------------

IMAS-MUSCLE3 registers its actors with MUSCLE3 through a Python
`entry point <https://muscle3.readthedocs.io/en/latest/models_as_software.html#python-package-entry-points>`_.
After installing the package with ``pip install imas-muscle3``, the actors
can be imported in a yMMSL v0.2 file. This requires MUSCLE3 0.10.0 or newer.

.. code-block:: yaml

  ymmsl_version: v0.2
  imports:
    - from imas_muscle3.actors import implementation sink_component

Describing the actors manually
------------------------------

With yMMSL v0.1, all of the actors can be used by adding them in the ymmsl
file like:

.. code-block:: yaml

  implementations:
    sink_component:
      executable: python
      args: -u -m imas_muscle3.actors.sink_component

.. toctree::
  :caption: Actors
  :maxdepth: 1

  actor_sink_source
  actor_olc
  actor_accumulator
  actor_visualization
