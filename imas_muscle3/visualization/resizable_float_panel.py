from typing import Any

import panel as pn
import param

pn.extension("floatpanel")
# Circumvent issue with float panels not being able to resize plots properly:
# https://github.com/holoviz/panel/issues/6157


class ResizableFloatPanel(pn.layout.FloatPanel):
    current_w = param.Integer(default=500, bounds=(None, None))
    current_h = param.Integer(default=500, bounds=(None, None))

    def __init__(
        self,
        *objects: Any,
        height_offset: int = 60,
        width: int = 500,
        height: int = 500,
        **params: Any,
    ) -> None:
        self._content_col = pn.Column(
            *objects, sizing_mode="fixed", width=width, height=height
        )
        # Force initial size parameters
        params.setdefault("width", width)
        params.setdefault("height", height)
        params.setdefault("position", "center")
        super().__init__(  # type: ignore[no-untyped-call]
            self._content_col, **params
        )
        self._height_offset = height_offset
        pn.bind(
            self._sync_size,
            self.param.current_w,
            self.param.current_h,
            watch=True,
        )

    def _sync_size(self, w: int, h: int) -> None:
        if w and h:
            self._content_col.width = w
            self._content_col.height = max(h - self._height_offset, 0)

    _scripts = dict(pn.layout.FloatPanel._scripts)

    _scripts[
        "render"
    ] = """
        if (state.panel) {
          view.run_script('close')
        }
        var config = {
          headerTitle: data.name,
          content: float,
          theme: data.theme,
          id: 'jsPanel'+data.id,
          position: view.run_script('get_position'),
          contentSize: `500 500`,
          onstatuschange: function() {
            data.status = this.status;
            if (this.status === 'maximized' || this.status === 'normalized') {
                data.current_w = Math.ceil(this.offsetWidth);
                data.current_h = Math.ceil(this.offsetHeight);
            }
          },
          onbeforeclose: function() {
            data.status = 'closed';
            return true;
          },
          resizeit: {
            resize: function(panel, paneldata, event) {
              data.current_w = Math.ceil(paneldata.width);
              data.current_h = Math.ceil(paneldata.height);
            }
          }
        }
        if (data.contained) {
          config.container = view.container
        }
        config = {...config, ...data.config}
        state.panel = jsPanel.create(config);
        if (data.status !== 'normalized') {
          view.run_script('status')
        }
        state.resizeHandler = (event) => {
          if (event.panel === state.panel) {
            view.invalidate_layout()
          }
        }
        document.addEventListener(
          'jspanelresizestop',
          state.resizeHandler,
          false
          )
    """
