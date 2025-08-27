import numpy as np
import panel as pn
import holoviews as hv
import webbrowser
import threading
import time
import param

import imas
import hvplot.xarray

pn.extension()
hv.extension('bokeh')

class VisualizationActor(param.Parameterized):
  my_bool = param.Boolean(default=True, label='meep')
  eq = param.ClassSelector(class_=imas.ids_toplevel.IDSToplevel)

  def __init__(self, plot_func):
    super().__init__()
    self.port = 5006  # Change if needed
    self.server = None
    self.plot = None

    self.duration = 10
    self.iter = 0
    self.eq = None
    self.my_bool = False

    if plot_func is not None:
      self.plot_func = plot_func
    else:
      # self.plot_func = self.example_plot
      self.plot_func = self.beep_boop_plot
    # self.dynamic_panel = pn.panel(self.example_plot(None))
    self.dynamic_panel = hv.DynamicMap(self.plot_func)

  def test_run(self):
    self.start_server()
    # Start server in a separate thread so shutdown timer can run
    print('hi')
    for i in range(self.duration):
      self.update_plot(None)
      time.sleep(1)
    print('ho')
    self.server.stop()

  def test_run2(self):
    self.start_server()
    print('hi')
    # time.sleep(2)
    with imas.DBEntry(
        "imas:hdf5?path=/home/ITER/sanderm/gitrepos/pds/run/temp_data/torax_nice_1",
        "r",
    ) as entry:
        self.eq = entry.get_slice("equilibrium", 0, imas.ids_defs.CLOSEST_INTERP)
    # xrds = imas.util.to_xarray(eq)
    # def plot(dsname):
    #     print(dsname)
    #     if dsname:
    #         try:
    #             return xrds[dsname].hvplot.explorer()
    #         except Exception as exc:
    #             return str(exc)
    #     return ""
    # ds_selector = pn.widgets.Select(options=list(xrds.keys()))
    # self.dynamic_panel = pn.Column(
    #     ds_selector,
    #     pn.bind(plot, ds_selector),
    #     sizing_mode="stretch_both",
    # )

    # x = eq.time_slice[-1].profiles_1d.rho_tor_norm
    # y = eq.time_slice[-1].profiles_1d.gm2
    # plot = hv.Curve((x, y)).opts(width=600, height=400, title="equilibrium gm2")
    # self.dynamic_panel.object = plot

    print('ho')
    self.my_bool = not self.my_bool
    # for i in range(5):
    #   self.my_bool = not self.my_bool
    #   time.sleep(1)
    time.sleep(5)
    self.server.stop()

  # @pn.depends('eq')
  @pn.depends('my_bool')
  def beep_boop_plot(self):
    if self.eq is not None:
      x = self.eq.time_slice[-1].profiles_1d.rho_tor_norm
      y = self.eq.time_slice[-1].profiles_1d.gm2
      plot = hv.Curve((x, y)).opts(width=600, height=400, title="equilibrium gm2")
    else:
      plot = self.example_plot(None)
    # self.dynamic_panel.object = plot
    return plot.opts(framewise=True, responsive=True)

  def example_plot(self, my_input):
    # --- Create a simple sine plot ---
    x = np.linspace(0, 10, 500)
    y = np.sin(x + self.iter * 3.14159/15)
    plot = hv.Curve((x, y)).opts(width=600, height=400, title="Sine Wave")
    self.iter += 1
    return plot

  def start_server(self):
    self.server = pn.serve(
        self.dynamic_panel,
        port=self.port,
        address="0.0.0.0",
        show=False,
        threaded=True,
        start=True
    )
    self.open_browser()

  def update_plot(self, my_input):
    self.plot = self.plot_func(my_input)
    self.dynamic_panel.object = self.plot

  def open_browser(self):
      url = f"http://localhost:{self.port}"
      try:
          webbrowser.open(url)
      except Exception as e:
          print(f"Could not open browser automatically: {e}")
      print(f"Dashboard should be available at {url}")

if __name__ == "__main__":
  # VisualizationActor(None).test_run()
  VisualizationActor(None).test_run2()
