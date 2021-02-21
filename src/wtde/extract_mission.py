import time

import fire
import wtde
from wtde import EventHandler

from watchdog.observers import Observer

def watch(watch_dir='/tmp/wt_screens', archive_dir='/tmp/wt_archive', error_dir='/tmp/wt_archive/errors'):
    """Watch a directory and process files going into it"""
    observer = Observer()
    handler = EventHandler(watch_dir, archive_dir=archive_dir, error_dir=error_dir)

    print('Starting watch on {}, archive to {}, errors to {}'.format(watch_dir, archive_dir, error_dir))
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Stopping directory watch as requested...')
        observer.stop()
        observer.join()
    finally:
        if observer.is_alive():
            print('Stopping directory watch on unexpected event...')
            observer.stop()
            observer.join()

def extract(directory):
    image_list = wtde.validate_input(directory)
    # next call a function to give us which image is which
    category = wtde.determine_category(image_list)
    stats_image = wtde.find_stats_screen(image_list, category.name)
    results_image = wtde.find_results_screen(image_list)
    battle_message_image = wtde.find_battle_message_screen(image_list)
    header_img = wtde.header_image(stats_image, category)

    header_str = wtde.header_image_to_text(header_img)
    try:
      mission_results = {
          'game_category': category.name,
          'game_mode': wtde.game_mode(header_str),
          'map_name': wtde.map_name(header_str),
          'map_type': wtde.map_type(header_str),
          'w_l': wtde.w_or_l(header_str)}
    except ValueError as e:
        print(e)
        results_header_img.show()

    print(mission_results)

def retry(archive_dir='/tmp/wt_archive', error_dir='/tmp/wt_errors'):
    """Iterate through error directories and re-attempt to import the files

    Keyword Arguments:
    archive_dir -- the directory to put any successful parses into
    error_dir -- the top level directory to look for error archives. Expects subdirectories in the `archive_error` format
                    (which is a date in YYYYMMDDHHMM format, each directory having 3 images in it)
    """
    wtde.retry_errors(archive_dir, error_dir)


def main():
    fire.Fire()

if __name__ == '__main__':
    main()
