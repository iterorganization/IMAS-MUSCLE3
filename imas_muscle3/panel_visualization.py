import runpy
import types
import webbrowser

import holoviews as hv
import panel as pn
import param

pn.extension()
hv.extension("bokeh")


class VisualizationActor(param.Parameterized):
    ids_dict = param.Dict(default={})

    def __init__(self, plot_file_path, port):
        super().__init__()
        self.port = port
        self.server = None

        ns = runpy.run_path(plot_file_path)
        funcs = [v for v in ns.values() if isinstance(v, types.FunctionType)]

        dmaps = []
        for f in funcs:
            bound = pn.bind(f, ids_dict=self.param.ids_dict)
            dmaps.append(hv.DynamicMap(bound))

        self.dynamic_panel = pn.Column(*dmaps)
        self.start_server()

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
