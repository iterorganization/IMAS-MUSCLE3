import panel as pn

pn.extension("floatpanel")
w1 = pn.widgets.TextInput(name="Text:")
w2 = pn.widgets.FloatSlider(name="Slider")

floatpanel = pn.layout.FloatPanel(
    "asdf",
    name="Basic FloatPanel",
    contained=False,
    position="center",
)

floatpanel2 = pn.layout.FloatPanel(
    "Try dragging me around.",
    name="Free Floating FloatPanel",
    contained=False,
    position="center",
)

floatpanel3 = pn.layout.FloatPanel(
    "Try dragging me around.",
    name="Free Floating FloatPanel",
    contained=False,
    position="center",
)


pn.Column(
    "**Example: Basic `FloatPanel`**", floatpanel, floatpanel2, floatpanel3, height=250
).servable()
