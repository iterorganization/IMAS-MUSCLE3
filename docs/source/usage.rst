.. _`usage`:

Using IMAS-MUSCLE3
==================

  IMAS-MUSCLE3 helper actors can be used like any other MUSCLE3 actors.
  For a quick reminder on MUSCLE3 workflows, see the `documentation <https://muscle3.readthedocs.io/en/latest/index.html>`_.

  All of the actors can be used by adding them in the ymmsl file like:

  .. code-block:: bash

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
