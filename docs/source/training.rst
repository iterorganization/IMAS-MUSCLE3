.. _`training`:

********
Training
********

Recorder & Dashboard Training
==============================

In this training you will learn the following:

- Recording live workflow traffic to disk with the :ref:`recorder actor <actor_recorder>`
- Writing plot files (``State``/``Plotter`` classes) that both record and plot data
- Viewing recorded data, live or after the fact, with ``muscle_dashboard``
- Using automatic extraction and machine description loading

This training assumes you are familiar with the with MUSCLE3 workflows. If you are not,
please take a look at the `MUSCLE3 documentation <https://muscle3.readthedocs.io/en/latest/>`_.

There is also training material available for the ITER Pulse Design Simulation, which can also serve as
an introduction to IMAS-MUSCLE3 workflows, which is available
`here <https://sharepoint.iter.org/departments/POP/CM/IMDesign/Code%20Documentation/PDS/courses/basic_user_training.html>`_. Note that this requires an ITER account to view.

All examples require that you have an environment with IMAS-MUSCLE3 up and running.
If you do not have this yet, please have a look at the :ref:`installation instructions <installing>`.

.. important::
   For this training you will need access to a graphical environment to view the
   dashboard. If you are on SDCC, it is recommended to follow this training
   through the NoMachine client, and using chrome as your default browser (there have been
   issues when using firefox through NoMachine).

Exercise 1a: Setting Up Your First Recording
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      We will start by running the recorder actor for a simple example configuration.
      First, create a yMMSL configuration file that sets up a simple recording pipeline with:

      1. A :ref:`source actor <actor_source>` that sends the equilibrium IDS
      2. A :ref:`recorder actor <actor_recorder>` that receives and records the data.

      Use the following settings in the yMMSL:

      - Source URI: ``imas:hdf5?path=/home/ITER/blokhus/public/imasdb/ITER/4/666666/3/``
      - Config file: ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py``

      This config file is a simple plot file that only extracts and plots the plasma
      current of an equilibrium IDS over time.
      Run the MUSCLE pipeline, supplying the yMMSL file you made:

      .. code-block:: bash

         muscle_manager --start-all <yMMSL file>

      Once it's running (or after it finishes), point ``muscle_dashboard`` at the run
      folder it created:

      .. code-block:: bash

         m3dash <run folder>

      Open the recorder's tab in your browser. What do you see?

      .. hint::
         There are premade examples available that you can use, located in this
         directory: ``imas_muscle3/visualization/examples``. For this specific exercise, take a look at the example yMMSL file in ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.ymmsl``. If you want detailed information about the recorder actor, take a look
         at the :ref:`documentation <actor_recorder>`.

   .. md-tab-item:: Solution

      Create a yMMSL file with the following content:

      .. literalinclude:: ../../imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.ymmsl
         :language: yaml

      When you launch the muscle_manager and then ``muscle_dashboard`` on its run
      folder, the recorder's tab will show the plasma current plotted over time. If
      the run is still going, the plot keeps growing as new time slices are recorded;
      the "Occurrence" dropdown lets you step back through earlier outer-loop
      iterations once more than one exists.


Exercise 1b: Understanding the Basic Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      Now that you were able to run the recorder in the previous exercise, let's
      take a look under the hood to see what config file that you supplied actually does.
      We will take a look at the example plotting script located in ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py``

      Every plot file used by the recorder (that also wants to be viewable in the
      dashboard) defines the following two classes:

      1. ``State(BaseState)``: This class handles extracting and storing data from incoming IDSs.
      2. ``Plotter(BasePlotter)``: This class handles how to plot the extracted data in the ``State`` class.

      Take a look at the simple example plotting script below that you used in previous exercise
      to record and visualize the plasma current (Ip) from an equilibrium IDS over time.

      **File:** `imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py`

      .. literalinclude:: ../../imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py
         :language: python

      What does the ``extract`` method do in the State class?

      What does the ``get_dashboard`` method do in the Plotter class?

   .. md-tab-item:: Solution

      The ``State`` class **must** implement the ``extract(self, ids)`` method.
      The ``extract`` method for this example case:

      - handles every IDS the recorder receives on its S port, one at a time. So first it checks if the incoming IDS is an equilibrium IDS.
      - Extracts the plasma current of the time slice (``ids.time_slice[0].global_quantities.ip``) and
        its corresponding time value (``ids.time[0]``), and stores it in an Xarray dataset.
      - Either stores the first Xarray dataset entry in ``self.data`` or appends it to the existing Xarray dataset.

      The recorder builds a fresh ``State`` for every message it records, so this
      accumulation only spans one call; the recorder's own Zarr store does the
      accumulating across messages (see :ref:`actor_recorder`).

      The ``Plotter`` class **must** implement the ``get_dashboard(self)`` method.
      The ``get_dashboard`` method for this example case:

      - Gets called once when the dashboard opens this recorder's tab.
      - Uses `HoloViews <https://holoviews.org/>`_ as its cornerstone to enable interactive visualizations.
      - Returns a `HoloViews DynamicMap <https://holoviews.org/reference/containers/bokeh/DynamicMap.html>`_ object,
        which allows you to dynamically update a plot whenever its argument function is called, here ``self.plot_ip_vs_time``.
      - Implements ``self.plot_ip_vs_time`` which automatically runs whenever the ``self.time`` parameter is updated.
        This happens when the dashboard picks up newly recorded data, or when the user changes the time slider in the UI.
        ``self.plot_ip_vs_time`` loads the state defined in the ``State`` class above, using ``self.active_state.data.get("equilibrium")``.
      - Extracts the Ip and time arrays from the state object, based on the selected time parameter.
      - It plots the plasma current versus time using a `HoloViews Curve <https://holoviews.org/reference/elements/bokeh/Curve.html>`_,
        which it returns to the DynamicMap, which will automatically update the plot.


Exercise 1c: Creating a custom plot file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      Now that you understand how the ``State`` and ``Plotter`` classes work, let's
      try to create your own plot file for the recorder. In this
      exercise you will learn how to visualize a 1D ff' profile, as a function of the
      poloidal flux, over time.

      For this exercise you can use the template below, in which you only have to implement
      the ``extract_equilibrium`` and ``plot_f_df_dpsi_profile`` methods.

      .. code-block:: python

         import holoviews as hv
         import numpy as np
         import param
         import xarray as xr

         from imas_muscle3.visualization.base_plotter import BasePlotter
         from imas_muscle3.visualization.base_state import BaseState


         class State(BaseState):
             def extract(self, ids):
                 if ids.metadata.name == "equilibrium":
                     self.extract_equilibrium(ids)

             def extract_equilibrium(self, ids):
                 # Implement this method!

         class Plotter(BasePlotter):
             def get_dashboard(self):
                 profile_plot = hv.DynamicMap(self.plot_f_df_dpsi_profile)
                 return profile_plot

             @param.depends("time")
             def plot_f_df_dpsi_profile(self):
                 # Implement this method!

      Implement the ``extract_equilibrium`` method which does the following:

      - Loads the ff' profile from the IDS: ``ids.time_slice[0].profiles_1d.f_df_dpsi``
      - Loads the corresponding psi coordinates: ``ids.time_slice[0].profiles_1d.psi``
      - Stores both in an Xarray Dataset.
      - Either saves the first entry in ``self.data`` or concatenates it to an existing Dataset.

      .. hint::
         Profile data is a 1D array for each time slice, so you'll need a dimension for
         the profile points in addition to time.

      Also implement the ``plot_f_df_dpsi_profile`` method in the ``Plotter`` class that
      displays the ff' profile stored in the state object as a function of psi for the current time step.

      Your ``plot_f_df_dpsi_profile`` should do the following:

      - Load the state data from the current ``self.active_state``.
      - Extract the arrays for ff' and psi from the state data (use ``state.sel(time=self.time)``).
      - Display psi on the x-axis and f_df_dpsi on the y-axis, using a `HoloViews Curve <https://holoviews.org/reference/elements/bokeh/Curve.html>`_.
      - Give an appropriate title, xlabel, and ylabel.
      - Properly handle the case when no data is available yet (Return an empty ``hv.Curve``).

      Point ``rec.config`` at your new file (reusing the yMMSL from exercise 1a) and
      view it with ``muscle_dashboard`` as before.

   .. md-tab-item:: Solution

      .. code-block:: python

         import holoviews as hv
         import numpy as np
         import param
         import xarray as xr

         from imas_muscle3.visualization.base_plotter import BasePlotter
         from imas_muscle3.visualization.base_state import BaseState


         class State(BaseState):
             def extract(self, ids):
                 if ids.metadata.name == "equilibrium":
                     self.extract_equilibrium(ids)

             def extract_equilibrium(self, ids):
                 ts = ids.time_slice[0]

                 profiles_data = xr.Dataset(
                     {
                         "f_df_dpsi": (("time", "profile"), [ts.profiles_1d.f_df_dpsi]),
                         "psi_profile": (("time", "profile"), [ts.profiles_1d.psi]),
                     },
                     coords={
                         "time": [ids.time[0]],
                         "profile": np.arange(len(ts.profiles_1d.f_df_dpsi)),
                     },
                 )

                 current_data = self.data.get("equilibrium")
                 if current_data is None:
                     self.data["equilibrium"] = profiles_data
                 else:
                     self.data["equilibrium"] = xr.concat(
                         [current_data, profiles_data], dim="time", join="outer"
                     )


         class Plotter(BasePlotter):
             def get_dashboard(self):
                 profile_plot = hv.DynamicMap(self.plot_f_df_dpsi_profile)
                 return profile_plot

             @param.depends("time")
             def plot_f_df_dpsi_profile(self):
                 xlabel = "Psi [Wb]"
                 ylabel = "ff'"
                 state = self.active_state.data.get("equilibrium")

                 if state:
                     selected_data = state.sel(time=self.time)
                     psi = selected_data.psi_profile.values
                     f_df_dpsi = selected_data.f_df_dpsi.values
                     title = f"ff' profile (t={self.time:.3f}s)"
                 else:
                     psi, f_df_dpsi, title = [], [], "Waiting for data..."

                 return hv.Curve((psi, f_df_dpsi), kdims=[xlabel], vdims=[ylabel]).opts(
                     framewise=True,
                     height=400,
                     width=600,
                     title=title,
                     xlabel=xlabel,
                     ylabel=ylabel,
                 )

      The recorder tab in ``muscle_dashboard`` now shows the ff' profile evolving
      over time instead of the Ip curve.


.. tip:: More complex examples of plot files are available in the
   ``imas_muscle3/visualization/examples/`` directory. For example, the PDS example
   combines data from multiple IDSs, handles machine description data, and
   handles different types of plots.


Exercise 2: Using Automatic Extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      In this exercise you will update your yMMSL configuration to enable automatic
      extraction. This lets the recorder discover and record time-dependent
      quantities without needing a custom ``extract`` method.

      Advantages of automatic extraction:

      - Useful for exploring unfamiliar datasets
      - Automatically discovers time-dependent quantities in the IDS
      - No need to manually write extraction code

      Disadvantages:

      - No fine grain control over what gets extracted
      - Unable to combine or derive data across quantities
      - Recording everything discoverable can be slow and memory-heavy

      Repeat exercise 1a, but this time use a bare config file and add the
      following settings to the yMMSL:

      .. code-block:: python

         from imas_muscle3.visualization.base_state import BaseState


         class State(BaseState):
             pass  # extraction is filled in automatically

      .. code-block:: yaml

         settings:
           rec.automatic_extract: true
           rec.automatic_extract_fields: >-
             equilibrium.time_slice[0].profiles_1d.dpressure_dpsi
             equilibrium.time_slice[0].global_quantities.energy_mhd

      Run the MUSCLE pipeline, supplying the yMMSL file you made, and inspect the
      recorded Zarr store (e.g. with ``zarr.open_group``) or add a ``Plotter`` to
      the config file to view the fields you listed.

   .. md-tab-item:: Solution

      With ``automatic_extract_fields`` unset, everything discoverable in the
      connected IDSs would be recorded; naming the two fields above restricts it
      to just the pressure-gradient profile and the MHD energy scalar. See
      :ref:`actor_recorder`'s "Automatic extraction" section for how the dotted
      field paths map onto the recorded Zarr groups.

Exercise 3: Viewing a Finished Run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      The recorder writes its data to disk independently of whether anything is
      watching it live. Run the yMMSL from exercise 1a to completion *without*
      starting ``muscle_dashboard``, then open the dashboard on the finished run's
      folder afterwards.

      .. hint::
         The recorder's Zarr stores live under
         ``<run folder>/instances/rec/workdir/<port>/``. ``muscle_dashboard``
         discovers them from that on-disk footprint alone, so it works the same
         whether the run is still going, finished cleanly, or crashed partway
         through.

   .. md-tab-item:: Solution

      .. code-block:: bash

         muscle_manager --start-all simple_1d_plot.ymmsl
         # ... wait for it to finish ...
         muscle_dashboard <run folder>

      The Ip curve appears exactly as it would have live, built from the
      completed Zarr store instead of an active MUSCLE3 connection.


Exercise 4: Loading Machine Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      In this exercise you will create a 2D plot that combines **static machine description data** with **time-evolving equilibrium data**. Specifically,

      - The plasma boundary outline (from the `equilibrium` IDS) as it changes over time.
      - The tokamak first wall and divertor (from the `wall` machine description IDS).

      Unlike the live traffic on the recorder's ``S`` port, machine description
      isn't streamed through MUSCLE3 at all: it's loaded on demand, straight from
      an IMAS URI, when a dashboard opens the recorder's tab. Add
      ``imas_muscle3.visualization.md_loader.init_state`` to your config file and
      point the recorder's ``md`` setting at it:

      - Add ``rec.md: wall=imas:hdf5?path=/home/ITER/blokhus/public/imasdb/ITER/4/666666/3/``
        to your yMMSL settings.
      - Import ``init_state`` at module level in your config file so the
        dashboard picks it up: ``from imas_muscle3.visualization.md_loader import init_state``.
      - Inside the ``Plotter``, ``self.active_state.md.get("wall")`` then gives
        you the loaded ``wall`` IDS.

      You will need to implement the ``extract_equilibrium``, ``_plot_boundary_outline``, and ``_plot_wall`` methods, in the template below.

      Implement the ``extract_equilibrium`` method which does the following:

      - Loads R and Z coordinates of the the boundary outline from the equilibrium IDS:
        ``ids.time_slice[0].boundary.outline.r``, ``ids.time_slice[0].boundary.outline.z``
      - Stores both in an Xarray Dataset.
      - Either saves the first entry in ``self.data`` or concatenates it to an existing Dataset.

      Implement the ``_plot_boundary_outline`` method which does the following:

      - Load the state data from the current ``self.active_state``.
      - Extract the r and z arrays from the state data (use ``state.sel(time=self.time)``).
      - Display r and z, using a `HoloViews Curve <https://holoviews.org/reference/elements/bokeh/Curve.html>`_.

      Implement the ``_plot_wall`` method which does the following:

      - Load the wall machine description IDS using ``self.active_state.md.get("wall")``.
      - Loads the first wall and divertor from the wall IDS:
        ``ids.description_2d[0].limiter.unit[i].outline.r`` and ``ids.description_2d[0].limiter.unit[i].outline.z`` for ``i = 0`` and ``i = 1``.
      - Display r and z, using a `HoloViews Path <https://holoviews.org/reference/elements/bokeh/Path.html>`_.

      .. code-block:: python

         import holoviews as hv
         import numpy as np
         import panel as pn
         import xarray as xr

         from imas_muscle3.visualization.base_plotter import BasePlotter
         from imas_muscle3.visualization.base_state import BaseState
         from imas_muscle3.visualization.md_loader import init_state  # noqa: F401


         class State(BaseState):
             def extract(self, ids):
                 if ids.metadata.name == "equilibrium":
                     self.extract_equilibrium(ids)

             def extract_equilibrium(self, ids):
                 # Implement this method!

         class Plotter(BasePlotter):
             DEFAULT_OPTS = hv.opts.Overlay(
                 xlim=(0, 13),
                 ylim=(-10, 10),
                 title="Wall and equilibrium boundary outline",
                 xlabel="r [m]",
                 ylabel="z [m]",
             )

             def get_dashboard(self):
                 elements = [
                     hv.DynamicMap(self._plot_boundary_outline),
                     hv.DynamicMap(self._plot_wall),
                 ]
                 overlay = hv.Overlay(elements).collate().opts(self.DEFAULT_OPTS)
                 return pn.pane.HoloViews(overlay, width=800, height=1000)

             @pn.depends("time")
             def _plot_boundary_outline(self):
                 # Implement this method!

             def _plot_wall(self):
                 # Implement this method!


   .. md-tab-item:: Solution

      Example yMMSL file:

      .. literalinclude:: ../../imas_muscle3/visualization/examples/machine_description/machine_description.ymmsl

      Example plotting script:

      .. literalinclude:: ../../imas_muscle3/visualization/examples/machine_description/machine_description.py
         :language: python
