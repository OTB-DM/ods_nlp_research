from typing import List, Dict, Tuple, Any


class DatasetReader:
    """An abstract class for reading data from some location and construction of a dataset."""

    def read(self, data_path: str, *args, **kwargs) -> Dict[str, List[Tuple[Any, Any]]]:
        """Reads a file from a path and returns data as a list of tuples of inputs and correct outputs
         for every data type in ``train``, ``valid`` and ``test``.
        """
        raise NotImplementedError