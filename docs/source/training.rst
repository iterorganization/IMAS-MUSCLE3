
.. _`training`:

********
Training
********

Visualization Actor Training
============================

In this training you will learn the following:

- Working with live data visualization
- Creating custom plotting scripts for the visualization actor
- Setting up visualization components in yMMSL files
- Using both MUSCLE3 and standalone modes

This training assumes you are familiar with the with MUSCLE3 workflows. If you are not,
please take a look at the `MUSCLE3 documentation <https://muscle3.readthedocs.io/en/latest/>`_.

There is also training material available for the ITER Pulse Design Simulation, which can also serve as
an introduction to IMAS-MUSCLE3 workflows, which is available 
`here <https://sharepoint.iter.org/departments/POP/CM/IMDesign/Code Documentation/PDS/courses/basic_user_training.html>`_. Note that this requires an ITER account to view.

All examples require that you have an environment with IMAS-MUSCLE3 up and running.
If you do not have this yet, please have a look at the :ref:`installation instructions <installing>`.

.. important::
   For this training you will need access to a graphical environment to visualize
   the simulation results. If you are on SDCC, it is recommended to follow this training
   through the NoMachine client, and using chrome as your default browser (there have been
   issues when using firefox through NoMachine).

Exercise 1a: Setting Up Your First Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      We will start by running the visualization actor for a simple example configuration. 
      First, create a yMMSL configuration file that sets up a simple visualization pipeline with:

      1. A :ref:`source actor <actor_sink_source>` that sends the equilibrium IDS
      2. A :ref:`visualization actor <actor_visualization>` that receives and plots the data.

      Use the following settings in the yMMSL:
      
      - Source URI: ``imas:hdf5?path=/home/ITER/blokhus/public/imasdb/ITER/4/666666/3/``
      - Plotting script: ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py``

      This plotting script is a simple configuration that will only plot the plasma current 
      of an equilibrium IDS over time. 
      Run the MUSCLE pipeline, supplying the yMMSL file you made:
      
      .. code-block:: bash
        
         muscle_manager --start-all <yMMSL file>

      The visualization actor will automatically open your default browser after it is initiated.
      What do you see in your browser?

      .. hint::
         There are premade examples available that you use can use available, located in this
         location: ``imas_muscle3/visualization/examples``. For this specific exercise, take a look at the example yMMSL file in ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.ymmsl``. If you want detailed information about the visualization actor, take a look
         at the :ref:`documentation <actor_visualization>`.

   .. md-tab-item:: Solution

      Create a file called ``my_visualization.ymmsl`` with the following content:

      .. code-block:: yaml

         ymmsl_version: v0.1
         model:
           name: my_visualization
           components:
             source_component:
               implementation: source_component
               ports:
                 o_i: [equilibrium_out]
             visualization_component:
               implementation: visualization_component
               ports:
                 s: [equilibrium_in]
           conduits:
             source_component.equilibrium_out: visualization_component.equilibrium_in
         settings:
           source_component.source_uri: imas:hdf5?path=/home/ITER/blokhus/public/imasdb/ITER/4/666666/3/
           visualization_component.plot_file_path: <path/to/IMAS-MUSCLE3>/imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py
         implementations:
           visualization_component:
             executable: python
             args: -u -m imas_muscle3.actors.visualization_component
           source_component:
             executable: python
             args: -u -m imas_muscle3.actors.source_component
         resources:
           source_component:
             threads: 1
           visualization_component:
             threads: 1

      When you launch the muscle_manager, the browser should open, and you will see the
      plasma current plotted over time, updating in real-time as the new time slices are 
      received by the visualization actor.

      .. figure:: ../source/images/ip_curve.gif


Exercise 1b: Understanding the Basic Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      Now that you were able to run the visualization actor in the previous exercise, let's
      take a look under the hood to see what the plotting script actually does.

      Every plotting script for the visualization actor must include the following two classes:

      1. ``State(BaseState)``: This class handles extracting and storing data from incoming IDSs.
      2. ``Plotter(BasePlotter)``: This class handles how to plot the extracted data in the ``State`` class.

      Take a look at the simple example plotting script below that you used in previous exercise 
      to visualize the plasma current (Ip) from an equilibrium IDS over time.

      **File:** `imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py`

      .. literalinclude:: ../../imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py
         :language: python

      What does the ``extract`` method do in the State class?
      
      What does the ``get_dashboard`` method do in the Plotter class?

   .. md-tab-item:: Solution

      The ``State`` class **must** implement the ``extract(self, ids)`` method.
      The ``extract`` method for this example case:
      
      - handles every IDS that is received on the S port, one at a time. So first it checks if the incoming IDS is an equilibrium IDS.
      - Extracts the plasma current of the time slice (``ids.time_slice[0].global_quantities.ip``) and 
        its corresponding time value (``ids.time[0]``), and stores it in an Xarray dataset.
      - Either stores the first Xarray dataset entry in ``self.data`` or appends it to the existing Xarray dataset.
      
      The ``Plotter`` class **must** implement the ``get_dashboard(self)`` method.
      The ``get_dashboard`` method for this example case:
      
      - Gets called once when the visualization actor is initialized.
      - Uses `HoloViews <https://holoviews.org/>`_ as its cornerstone to enable interactive visualizations.
      - Returns a `HoloViews DynamicMap <https://holoviews.org/reference/containers/bokeh/DynamicMap.html>`_ object, 
        which allows you to dynamically update a plot whenever its argument function is called, here ``self.plot_ip_vs_time``.
      - Implements ``self.plot_ip_vs_time`` which automatically runs whenever the ``self.time`` parameter is updated. 
        This happens when the Visualization actor receives new data, or when the user changes the time slider in the UI.
        ``self.plot_ip_vs_time`` loads the state defined in the ``State`` class above, using ``self.active_data.data.get("equilibrium")``.
      - Extracts the Ip and time arrays from the state object, based on the selected time parameter.
      - It plots the plasma current versus time using a `HoloViews Curve <https://holoviews.org/reference/elements/bokeh/Curve.html>`_,
        which it returns to the DynamicMap, which will automatically update the plot.


Exercise 1c: Creating a custom visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      Now that you understand how the ``State`` and ``Plotter`` classes work, let's
      try to create your own plotting script for the visualization actor. In this 
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

      This generates the following ff' plot over time:

      .. figure:: ../source/images/ff_prime.gif

Exercise 1d: Loading Machine Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tip:: More complex examples of visualizations are available in the 
   ``imas_muscle3/visualization/examples/`` directory. For example, the PDS example
   combines data from multiple IDSs, handles machine description data, and 
   handles different types of plots.


Exercise 2: Using Automatic Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      In this exercise you will your yMMSL configuration to enable automatic mode. This mode allows
      the visualization actor to automatically discover and plot time-dependent 
      quantities without needing a custom plotting script.

      Advantages of automatic mode:
      
      - Useful for exploring unfamiliar datasets
      - Automatically discovers all time-dependent quantities in the IDS
      - Provides a dropdown menu to select quantities to visualize
      - Chooses appropriate plot types automatically
      - No need to manually extract quantities

      Disadvantages:

      - No fine grain control over the plots
      - Unable to combine data
      - Slower performance and increased memory usage

      Repeat exercise 1a, however this time add the following settings to the yMMSL:

      .. code-block:: yaml

         settings:
           visualization_component.automatic_mode: true
           visualization_component.automatic_extract_all: true

      Run the MUSCLE pipeline, supplying the yMMSL file you made. Use the dropdown menu to 
      visualize the following parameters:

      - ``equilibrium/time_slice[0]/profiles_1d[0]/dpressure_dpsi``
      - ``equilibrium/time_slice[0]/global_quantities/energy_mhd``

   .. md-tab-item:: Solution

      Besides the plasma current curve, which was defined in the plotter class, you 
      should also see the p' and the MHD energy curves in separate panels:

      .. figure:: ../source/images/automatic.png

Exercise 3: Using the CLI
^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      It is also possible to run the visualization actor from the command line instead,
      without setting up a MUSCLE3 workflow. Try running the simple_1d_plot example 
      through the CLI.

      Run the visualization with:
      
      - URI: ``imas:hdf5?path=/home/ITER/blokhus/public/imasdb/ITER/4/666666/3/``
      - Plotting script: ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py``

      .. hint::
         Use ``python -m imas_muscle3.visualization.cli --help`` to see available options.

   .. md-tab-item:: Solution

      Run the following command:

      .. code-block:: bash

         python -m imas_muscle3.visualization.cli \
             "imas:hdf5?path=/home/ITER/blokhus/public/imasdb/ITER/4/666666/3/" \
             imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py

