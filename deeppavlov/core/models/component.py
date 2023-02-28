from abc import ABCMeta, abstractmethod

from logging import getLogger

log = getLogger(__name__)


class Component(metaclass=ABCMeta):
    """Abstract class for all callables that could be used in Chainer's pipe."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def destroy(self):
        attr_list = list(self.__dict__.keys())
        for attr_name in attr_list:
            attr = getattr(self, attr_name)
            if hasattr(attr, 'destroy'):
                attr.destroy()
            delattr(self, attr_name)