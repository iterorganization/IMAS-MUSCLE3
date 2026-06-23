"""Read-side viewer for distilled Zarr stores.

A small, imas-aware plotting layer over the Zarr stores written by the
:mod:`~imas_muscle3.actors.distill_component` recorder. It is exposed to the
generic ``muscle3-dashboard`` as a *run panel* plugin (entry-point group
``muscle3_dashboard.run_panels``) via :func:`imas_muscle3.viewer.panel_app.make_panel`,
and can also be served standalone.
"""
