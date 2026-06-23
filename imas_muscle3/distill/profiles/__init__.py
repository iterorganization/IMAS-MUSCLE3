"""Reference visualization profiles (paired custom distiller + bespoke view).

Each module here is a self-contained example of the profile contract documented
in :mod:`imas_muscle3.viewer.profile`: an ``extract(ids)`` half (run by the
distill recorder via its ``config`` setting) and a ``plot(data, time_index)``
half (run by the viewer). They double as templates for workflow-specific
profiles.
"""
