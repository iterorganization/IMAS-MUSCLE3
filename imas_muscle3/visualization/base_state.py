import abc

import param
import xarray as xr


class BaseState(param.Parameterized):
    data = param.ClassSelector(class_=xr.Dataset, default=xr.Dataset())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_data()

    @abc.abstractmethod
    def _initialize_data(self):
        pass

    @abc.abstractmethod
    def update(self, ids):
        pass
