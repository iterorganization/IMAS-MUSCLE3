.. _`actor_visualization`:

Visualization actor
===================

Actor for live, web-based visualization of IMAS data in a simulation using the 
`Panel <https://panel.holoviz.org/>`_ library.
It launches a local web server to display the visualization.
The plotting logic is user-defined in a specified Python script, allowing for customizable visualizations.

To use the visualization actor, include it in the ymmsl file:

.. code-block:: bash

  implementations:
    visualization_component:
      executable: python
      args: -u -m imas_muscle3.actors.visualization_component

.. warning::
    The visualization actor is still in a prototype state and may not always behave as expected!

Available Settings
------------------

* Mandatory

  - ***plot_file_path***: (string) The path to the Python script that defines the plotting logic. This script must contain a `State` class and a `Plotter` class, as described below.

* Optional

  - ***port***: (int) The port on which the visualization server will run. Defaults to `5006`.
  - ***throttle_interval***: (float) The minimum time in seconds between plot updates. 
    This can be used to prevent the visualization from slowing down if data arrives very quickly. Defaults to `0.1`.
  - ***keep_alive***: (bool) If `True`, the visualization server will remain active after 
    the last MUSCLE message was received, allowing for inspection of the received data. 
    If `False`, the server stops after the last MUSCLE message is received. Defaults to `False`.
  - ***open_browser***: (bool) If `True`, automatically opens a new tab in your web 
    browser to the visualization page upon starting. Defaults to `True`.
  - ***automatic_mode***: (bool) If `True`, time-dependent IDS quantities can automatically
    be visualized using a dropdown menu. Data will automatically be extracted, and an
    appropriate plot for the quantity will be selected.
  - ***automatic_extract_all***: (bool) This only works if automatic_mode is enabled. 
    If `True`, automatically extracts all time dependent quantities of the IDS. 
    If `False`, the data will only start to be extracted as soon as a new plot is selected. 
    Enabling this can cause significant computation and memory overhead, depending on the size of the IDS.

Available Ports
---------------

All IDS's are available for the visualization actor. They will be active if connected 
in the ymmsl file and will be skipped otherwise.

* Optional

  - ***<ids_name>_in (S)***: Any incoming IDS timeslices on the `S` port. 
    Replace `<ids_name>` with the required IDS name (e.g., ``equilibrium_in``).
  - ***<ids_name>_md_in (F_INIT)***: Any incoming machine description IDS's on the `F_INIT` port. 
    These are typically static data like the machine wall or coil geometry. 
    Replace `<ids_name>` with the required IDS name (e.g., ``wall_md_in``).

User-defined Plotting Script
----------------------------

The Python script specified by `plot_file_path` is the core of the visualization. 
It must define two classes that inherit from the provided base classes:

1.  **State(BaseState)**: This class is responsible for extracting the necessary information 
    from an IDS into an internal data structure, typically an `xarray.Dataset`. 
    It must implement an ``extract(self, ids)`` method which will be called for each incoming IDS.

2.  **Plotter(BasePlotter)**: This class uses the data managed by the `State` object to 
    define and arrange the plots in a Panel dashboard. It must implement a 
    ``get_dashboard(self)`` method that returns a Panel object.

Command Line Interface
----------------------

A command line interface is made available if you want to run the visualization actor
directly from the terminal instead of through a MUSCLE3 actor. Usage details can be 
found by supplying the ``--help`` option:

.. code-block:: bash

  python ./imas_muscle3/visualization/cli.py --help

Example
--------

This example demonstrates how to set up a simple live plot of a single variable:
the plasma current (Ip) of an equilibrium IDS. More complex examples are available in
the ``imas_muscle3/visualization/examples`` directory.


ymmsl configuration
^^^^^^^^^^^^^^^^^^^

This configuration consists of two main components:

- A :ref:`source actor <actor_sink_source>` is set up to send an equilibrium IDS.
- A visualization actor is set up to receive this IDS and visualize it.

.. literalinclude:: ../../imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.ymmsl
   :language: yaml
   :caption: simple_1d_plot.ymmsl

Plotting Script
^^^^^^^^^^^^^^^


This script tells the visualization actor **what** to extract from the data and **how** to plot it. 

**State Class**

The State class extracts the plasma current from the equilibrium IDS. 
When it receives a new equilibrium IDS time slice, it extracts out the plasma current
(``ts.global_quantities.ip``) and time, then stores this in an xarray Dataset for accumulation over time.

**Plotter Class**

The Plotter class creates a live-updating line plot. It returns a
`HoloViews DynamicMap <https://holoviews.org/reference/containers/bokeh/DynamicMap.html>`_
object that automatically calls ``plot_ip_vs_time()`` when data updates.
This method plots all plasma current data received up to the current time as a time series curve.

.. literalinclude:: ../../imas_muscle3/visualization/examples/simple_1d_plot/simple_1d_plot.py
   :language: python
   :caption: simple_1d_plot.py

General
-------
The visualization actor is not bound to a specific DD version.
