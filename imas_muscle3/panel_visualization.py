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
        self.port = 5006  # Change if needed
        self.server = None
        self.plot = None

        self.duration = 10
        self.iter = 0
        self.eq = None
        self.ip_history = []
        self.time_history = []

        if plot_func is not None:
            self.plot_func = plot_func
        else:
            self.plot_func = self.eq_plot
        self.dynamic_panel = hv.DynamicMap(self.plot_func)

    @pn.depends("eq")
    def eq_plot(self):
        print("plotting eq")
        if self.eq:
            ts = self.eq.time_slice[0]

            time = self.eq.time[0]
            self.ip_history.append(ts.global_quantities.ip)
            self.time_history.append(time)

            curve = hv.Curve(
                (self.time_history, self.ip_history), "Time (s)", "Ip (A)"
            ).opts(self.OPTIONS)
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
        self.open_browser()

    def stop_server(self):
        self.server.stop()

    def open_browser(self):
        url = f"http://localhost:{self.port}"
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
        print(f"Dashboard should be available at {url}")
