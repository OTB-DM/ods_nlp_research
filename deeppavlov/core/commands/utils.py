from pathlib import Path
from typing import Union


def expand_path(path: Union[str, Path]) -> Path:
    """Convert relative paths to absolute with resolving user directory."""
    return Path(path).expanduser().resolve()