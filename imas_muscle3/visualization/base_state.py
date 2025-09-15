import param


class BaseState(param.Parameterized):
    data = param.Dict(default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, ids):
        raise NotImplementedError(
            "a state class needs to implement an `extract` method"
        )


class BasePlotter(param.Parameterized):
    state = param.Parameter()

    def __init__(self, state, **params):
        super().__init__(state=state, **params)

    def get_dashboard(self):
        raise NotImplementedError(
            "a state class needs to implement a `get_dashboard` method"
        )
