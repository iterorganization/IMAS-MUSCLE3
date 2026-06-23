"""Visualization *profiles*: paired custom distiller + bespoke view.

A profile is a single Python module describing one workflow's bespoke
visualization, with two halves used at opposite ends of the pipeline:

* ``extract(ids) -> dict[str, xarray.Dataset]`` — the **custom distiller**,
  loaded by :mod:`~imas_muscle3.actors.distill_component`. It computes derived,
  non-IMAS quantities (separatrix, contours, flux-surface averages, …) from the
  raw IDS and returns them as single-time datasets to record.
* ``plot(data, time_index) -> Viewable`` — the **bespoke view**, loaded by the
  viewer. Given a :class:`ProfileData` reader over the run's stores and a
  selected time index, it returns a Panel/holoviews view (an overlay of the
  separatrix and contours, profile panels, time series, …). The viewer wraps it
  with a time slider and live-tail polling, so a profile author only writes
  "given the data at time *t*, draw this".

Either half may be omitted: a profile with only ``extract`` records custom data
that the generic browser still plots variable-by-variable; one with only
``plot`` re-renders auto-distilled data bespokely.

The recorder stamps the profile's path into each store's root attrs, so the
viewer can find and load the matching ``plot`` (see
:func:`imas_muscle3.viewer.store.store_profile`).
"""

import logging
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import xarray as xr

from imas_muscle3.distill.zarr_sink import group_name
from imas_muscle3.viewer import store as store_mod

logger = logging.getLogger(__name__)


def load_profile(path: str) -> SimpleNamespace:
    """Load a profile module from ``path``, exposing its ``extract``/``plot``.

    Raises if the file defines neither, so a misconfigured profile fails loudly
    rather than silently rendering nothing.
    """
    namespace = runpy.run_path(path)
    extract = namespace.get("extract")
    plot = namespace.get("plot")
    if extract is None and plot is None:
        raise NameError(
            f"{path} is not a visualization profile: it defines neither "
            f"'extract(ids)' nor 'plot(data, time_index)'."
        )
    return SimpleNamespace(extract=extract, plot=plot, path=path)


class ProfileData:
    """Lazily read distilled groups across one run's stores, for ``plot``.

    Profiles address data by group name (e.g. ``"derived/separatrix"``); a group
    is looked up across every store of the run, or in a named store. Opened
    datasets are cached for one render; :meth:`reset` drops the cache so the next
    render (a live-tail tick) re-reads the growing stores.
    """

    def __init__(self, stores: Dict[str, Path]) -> None:
        self._stores = stores
        self._cache: Dict[tuple, Optional[xr.Dataset]] = {}

    def stores(self) -> List[str]:
        """Labels (port stems) of the run's stores."""
        return sorted(self._stores)

    def dataset(
        self, group: str, store: Optional[str] = None
    ) -> Optional[xr.Dataset]:
        """Open a distilled group (all times), or ``None`` if not present.

        With ``store`` given, look only in that store; otherwise return the
        first store that has the group. ``group`` may be the logical key used
        when distilling (``"derived/separatrix"``); it is mapped to the on-disk
        group name automatically.
        """
        disk = group_name(group)
        key = (store, disk)
        if key in self._cache:
            return self._cache[key]
        labels = [store] if store is not None else self.stores()
        result: Optional[xr.Dataset] = None
        for label in labels:
            path = self._stores.get(label)
            if path is None:
                continue
            try:
                if disk in store_mod.list_groups(path):
                    result = store_mod.open_group(path, disk)
                    break
            except Exception:
                logger.warning(
                    "failed reading %s/%s", path, group, exc_info=True
                )
        self._cache[key] = result
        return result

    def n_times(self) -> int:
        """Largest time length across all groups in all stores."""
        n = 0
        for label in self.stores():
            for group in store_mod.list_groups(self._stores[label]):
                ds = self.dataset(group, store=label)
                if ds is not None:
                    n = max(n, ds.sizes.get(store_mod.TIME, 0))
        return n

    def reset(self) -> None:
        """Drop cached datasets so the next access re-reads the stores."""
        self._cache.clear()
