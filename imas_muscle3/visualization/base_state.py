import abc

import param
import xarray as xr


class BaseState(param.Parameterized):
    data = param.ClassSelector(class_=xr.Dataset)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def update(self, ids):
        pass


class BasePlotter(param.Parameterized):
    state = param.Parameter()

    def __init__(self, state, **params):
        super().__init__(state=state, **params)
