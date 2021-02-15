from typing import NamedTuple

class ReadyDir(NamedTuple):
    """Datastructure with all components needed to scrape scores"""

    files: list # files should be a list with 3 or more files in the watched directory
    directory: str # the watched directory. May consider a pathlib.Path at some point
    archive_dir: str # the directory to archive files to, when we have processed them
    error_dir: str # a directory to store errored sets that we couldn't figure out (yet)
