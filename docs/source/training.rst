
.. _`training`:

********
Training
********

Visualization Actor Training
============================

In this training you will learn the following:

- Working with live data visualization
- Creating custom plotting scripts for the visualization actor
- Setting up visualization components in YMMSL files
- Using both MUSCLE3 and standalone modes

All examples assume you have an environment with IMAS-MUSCLE3 up and running.
If you do not have this yet, please have a look at the :ref:`installation instructions <installing>`.

For this training you will need access to a graphical environment to visualize
the simulation results. If you are on SDCC, it is recommended to follow this training
through the NoMachine client, and using chrome as your default browser (there have been
issues when using firefox through NoMachine).



Exercise 1a: Understanding the Basic Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      Before creating visualizations, you need to understand the basic structure of a visualization script.
      Every visualization script must define two classes:

      1. **State(BaseState)**: Extracts data from incoming IDSs
      2. **Plotter(BasePlotter)**: Defines how to visualize the data

      Look at the simple example below that visualizes plasma current (Ip) from an equilibrium IDS:

      **File:** `imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py`

      .. literalinclude:: ../../imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py
         :language: python

      What does the ``extract`` method do in the State class?
      
      What does the ``get_dashboard`` method return in the Plotter class?

      .. hint::
         The State class processes incoming IDS data, while the Plotter class creates the visual display.

   .. md-tab-item:: Solution

      The ``extract`` method:
      
      - Checks if the incoming IDS is an equilibrium IDS
      - Extracts the plasma current (``ts.global_quantities.ip``) and time
      - Stores the data in an xarray Dataset for accumulation over time
      
      The ``get_dashboard`` method:
      
      - Returns a HoloViews DynamicMap object
      - The DynamicMap automatically updates when new data arrives
      - It displays a line plot of plasma current versus time

Exercise 1b: Setting Up Your First Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      You will now run the visualization actor for configuration of the previous exercise. First, 
      create a YMMSL configuration file that sets up a simple visualization pipeline with:

      1. A source actor that sends the equilibrium IDS
      2. A visualization actor that receives and plots the data, according to the 
         configuration in previous exercise.

      Use the following settings in the YMMSL:
      
      - Source URI: ``imas:hdf5?path=/home/ITER/fargerb/public/imasdb/ITER/3/666666/1``
      - Plot script: ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py``

      Run the MUSCLE pipeline, supplying the YMMSL file you made:
      
      .. code-block:: bash
        
         muscle_manager --start-all <YMMSL file>

      What do you see in your browser?

      .. hint::
         Look at the example YMMSL file in ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.ymmsl``

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
           source_component.source_uri: imas:hdf5?path=/home/ITER/fargerb/public/imasdb/ITER/3/666666/1
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

      When you launch the muscle_manger, the browser should open, and you will see the
      plasma current plotted over time, updating in real-time as the new time slices are 
      received by the visualization actor.

      .. figure:: ../source/images/ip_curve.gif

Exercise 1c: Extracting 1D Profile Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      Create a State class that extracts the ff' profile from an equilibrium IDS. 
      This profile data is stored as a 1D array along with its corresponding psi coordinate.

      Your State class should extract:
      
      - The ff' values: ``ts.profiles_1d.f_df_dpsi``
      - The psi coordinate: ``ts.profiles_1d.psi``
      - Store both in an xarray Dataset with dimensions ``("time", "profile")``

      .. hint::
         Profile data is 1D at each time slice, so you'll need a dimension for 
         the profile points in addition to time.

      Also create a Plotter class that displays the ff' profile as a function of psi
      for the current time step. The plot should show how the profile evolves as new 
      data arrives.

      Your plot should:
      
      - Display f_df_dpsi on the y-axis and psi on the x-axis
      - Show only the profile at the current time (use ``state.sel(time=self.time)``)
      - Update automatically when new data arrives (use ``@param.depends("time")``)

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
                     self._extract_equilibrium(ids)

             def _extract_equilibrium(self, ids):
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

.. tip:: More complex examples of visualizations are available in the 
   ``imas_muscle3/visualization/examples/`` directory. For example, the PDS example
   combines data from multiple IDSs, handles machine description data, and 
   visualizes 2-dimensional data.

Exercise 2: Using Automatic Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. md-tab-set::
   .. md-tab-item:: Exercise

      Modify your YMMSL configuration to enable automatic mode. This mode allows
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

      Repeat exercise 1b, however this time add the following settings to the YMMSL:

      .. code-block:: yaml

         settings:
           visualization_component.automatic_mode: true
           visualization_component.automatic_extract_all: true

      Run the MUSCLE pipeline, supplying the YMMSL file you made. Use dropdown menu to 
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
      
      - URI: ``imas:hdf5?path=/home/ITER/fargerb/public/imasdb/ITER/3/666666/1``
      - IDS name: ``equilibrium``
      - Plotting script: ``imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py``

      .. hint::
         Use ``python -m imas_muscle3.visualization.cli --help`` to see available options.

   .. md-tab-item:: Solution

      Run the following command:

      .. code-block:: bash

         python -m imas_muscle3.visualization.cli \
             "imas:hdf5?path=/home/ITER/fargerb/public/imasdb/ITER/3/666666/1" \
             equilibrium \
             imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py

