from typing import NamedTuple

class ReadyDir(NamedTuple):
    """Datastructure with all components needed to scrape scores"""
    # files should be a list with 3 or more files in the watched directory
    files: list
    # the watched directory. May consider a pathlib.Path at some point
    directory: str
