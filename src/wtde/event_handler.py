import os

import watchdog.events
from result import Ok, Err

import wtde

class EventHandler(watchdog.events.FileSystemEventHandler):
    """Handle changes to the watch directory"""

    def __init__(self, watch_dir):
        if not os.path.isdir(watch_dir):
            raise ValueError('{} must be a directory'.format(watch_dir))
        self.watch_dir = watch_dir

    def on_created(self, event: watchdog.events.FileCreatedEvent):
        """Dispatch a validated ReadyDir to wtde for scraping

        Positional Arguments:
        event -- A FileCreatedEvent

        Error:
        Should not directly raise an error
        """
        print('Oo! A file!')
        ready_dir = self._ready(event.src_path, self.watch_dir)
        if ready_dir.is_ok():
            wtde.handle_ready_files(ready_dir.value)
            return

        print(ready_dir.value)

    def _ready(self, file, watched_dir):
        """Validate that the files in the directory are ready for processing

        Positional Arguments:
        file -- full path to a file
        watched_dir -- the directory this event handler is watching

        Returns:
        A result.Ok containing a ReadyDir if we are ready, or a result.Err if not

        Error:
        Should only return result.Err. No expected raises
        """
        changed_dir = os.path.dirname(file)
        if changed_dir != watched_dir:
            return Err('File {} is not in watched directory {}. Ignoring'.format(file, watched_dir))
        dir_files = os.listdir(changed_dir)
        if len(dir_files) < 3:
            return Err('We expect at least 3 files to operate. Waiting')

        return Ok(wtde.ReadyDir(files=dir_files, directory=changed_dir))
