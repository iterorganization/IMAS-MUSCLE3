import webbrowser

import holoviews as hv
import imas
import panel as pn
import param

pn.extension()
hv.extension("bokeh")


class VisualizationActor(param.Parameterized):
    eq = param.ClassSelector(class_=imas.ids_toplevel.IDSToplevel)
    OPTIONS = hv.opts.Curve(framewise=True, responsive=True)

    def __init__(self, plot_func=None):
        super().__init__()
        self.port = 5006
        self.server = None
        self.plot = None

        self.duration = 10
        self.iter = 0
        self.eq = None

        if plot_func is not None:
            self.plot_func = plot_func
        else:
            self.plot_func = self._plot
        self.dynamic_panel = hv.DynamicMap(self.plot_func)

    @pn.depends("eq")
    def _plot(self):
        print("plotting eq")
        if self.eq:
            ts = self.eq.time_slice[0]
            f_df_dpsi = ts.profiles_1d.f_df_dpsi
            psi = ts.profiles_1d.psi
            curve = hv.Curve((psi, f_df_dpsi), "Psi", "ff'").opts(self.OPTIONS)
        else:
            curve = hv.Curve(([0, 1, 2], [0, 1, 2])).opts(self.OPTIONS)
        return curve

    def start_server(self):
        self.server = pn.serve(
            self.dynamic_panel,
            port=self.port,
            address="0.0.0.0",
            show=False,
            threaded=True,
            start=True,
        )
        self._open_browser()

    def stop_server(self):
        self.server.stop()

    def _open_browser(self):
        url = f"http://localhost:{self.port}"
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
        print(f"Dashboard should be available at {url}")
