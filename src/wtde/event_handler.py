import os
import time

import watchdog.events
from result import Ok, Err

import wtde

class EventHandler(watchdog.events.FileSystemEventHandler):
    """Handle changes to the watch directory"""

    def __init__(self, watch_dir, archive_dir='/tmp/wt_archive', error_dir='/tmp/wt_archive/errors'):
        if not os.path.isdir(watch_dir):
            raise ValueError('{} must be a directory'.format(watch_dir))
        self.watch_dir = watch_dir
        self.archive_dir = archive_dir
        self.error_dir = error_dir

    def on_created(self, event: watchdog.events.FileCreatedEvent):
        """Dispatch a validated ReadyDir to wtde for scraping

        Positional Arguments:
        event -- A FileCreatedEvent

        Error:
        Should not directly raise an error
        """
        ready_dir = self._ready(event.src_path, self.watch_dir, self.archive_dir, self.error_dir)
        if ready_dir.is_ok():
            # screenshot takes some time to finish after the file is created
            handle_free = wtde.utils.wait_for_no_handles(ready_dir.unwrap().files)
            if handle_free.is_ok():
                wtde.handle_ready_files(ready_dir.value)
            else:
                print(handle_free.unwrap())
            return

        print(ready_dir.value)

    def _ready(self, file, watched_dir, archive_dir, error_dir):
        """Validate that the files in the directory are ready for processing

        This is kept to my normal functional style because the only reason I use a class here is because
        of the library

        Positional Arguments:
        file -- full path to a file
        watched_dir -- the directory this event handler is watching
        archive_dir -- the directory to move processed files into
        error_dir -- the directory to move files to when we cannot handle them

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

        # TODO validate that filetypes are of images
        # TODO ignore non-image files

        return Ok(wtde.ReadyDir(files=dir_files, directory=changed_dir, archive_dir=archive_dir, error_dir=error_dir))
