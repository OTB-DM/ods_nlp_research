from abc import abstractmethod

from .component import Component
from .serializable import Serializable


class Estimator(Component, Serializable):
    """Abstract class for components that could be fitted on the data as a whole."""

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass