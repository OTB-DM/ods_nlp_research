from abc import ABCMeta, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Union, Optional

from deeppavlov.core.commands.utils import expand_path

log = getLogger(__name__)


class Serializable(metaclass=ABCMeta):
    """Abstract base class that expresses the interface for all models that can serialize data to a path."""

    def __init__(self, save_path: Optional[Union[str, Path]], load_path: Optional[Union[str, Path]] = None,
                 mode: str = 'infer',
                 *args, **kwargs) -> None:

        if save_path:
            self.save_path = expand_path(save_path)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = None

        if load_path:
            self.load_path = expand_path(load_path)
            if mode != 'train' and self.save_path and self.load_path != self.save_path:
                log.warning("Load path '{}' differs from save path '{}' in '{}' mode for {}."
                            .format(self.load_path, self.save_path, mode, self.__class__.__name__))
        elif mode != 'train' and self.save_path:
            self.load_path = self.save_path
            log.warning("No load path is set for {} in '{}' mode. Using save path instead"
                        .format(self.__class__.__name__, mode))
        else:
            self.load_path = None
            log.warning("No load path is set for {}!".format(self.__class__.__name__))

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass