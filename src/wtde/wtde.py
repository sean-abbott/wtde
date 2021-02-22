import imghdr
import os
import re
import shutil
import time

from datetime import datetime as dt
from enum import Enum
from typing import NamedTuple

import cv2
import numpy as np
import pytesseract

from matplotlib import pyplot as plt
from result import Ok, Err
from PIL import Image, UnidentifiedImageError

from wtde import utils

IS_DUALSCREEN = True

TEMPLATE_DIR = 'template_images'
TEMPLATE_PATHS = {
    'naval_mode': os.path.join(TEMPLATE_DIR, 'NavalMatch.png'),
    'results_badge': os.path.join(TEMPLATE_DIR, 'results_screen.png'),
    'naval_stats': os.path.join(TEMPLATE_DIR, 'naval_stats.png'),
    'ground_stats': os.path.join(TEMPLATE_DIR, 'ground_stats.png'),
    'air_stats': os.path.join(TEMPLATE_DIR, 'air_stats.png'),
    'battle_message_won_badge': os.path.join(TEMPLATE_DIR, 'battle_message_screen_won.png'),
    'battle_message_lost_badge': os.path.join(TEMPLATE_DIR, 'battle_message_screen_lost.png'),
}

# These are likely to need tweaking
TEMPLATE_THRESHOLDS = {
    'results_badge': 40000000,
    'naval_mode': 5000000,
    'naval_stats': 60000000,
    'air_stats': 50000000,
    'ground_stats': 60000000,
    'battle_message_won_badge': 15000000,
    'battle_message_lost_badge': 15000000,
}

GAMECATEGORY = Enum('GAMECATEGORY', 'ground air naval')
DT_FORMAT_STR = '%Y%m%d%H%M'

class ResultSet(NamedTuple):
    """Ongoing results built over the chain"""
    src_dir: str # the directory from which the images comes
    archive_dir: str # the top directory into which the images go when archive
    error_dir: str # the top directory into which errored sets go
    set_dir: str # the sub-directory under archive_dir or error_dir which this set goes in
    image_list: list # the list of PIL images we work with
    file_list: list # list of file(names/paths) we're working with
    stats_image: Image.Image # the PIL.image representing the stats page
    results_image: Image.Image # the PIL.image representing the results page
    battle_message_image: Image.Image # the PIL.image representing the battle messages page
    category: GAMECATEGORY # the category the of the mission
    game_map: str # the map name
    dt: str # TODO make datetime the date and approximate time of the mission
    remove_srcdir: bool # whether to remove the srcdir when archiving. used when retrying errors

class ReadyDir(NamedTuple):
    """Datastructure with all components needed to scrape scores"""

    files: list # files should be a list with 3 or more files in the watched directory
    directory: str # the watched directory. May consider a pathlib.Path at some point
    archive_dir: str # the directory to archive files to, when we have processed them
    error_dir: str # a directory to store errored sets that we couldn't figure out (yet)
    play_dt: dt # a datetime for when this match was played. Optional
    remove_srcdir: bool # whether to remove the srcdir when archiving. used when retrying errors

# TODO this is ugly AF. Find a better way
def update_result(
        r=None,
        src_dir=None,
        archive_dir=None,
        error_dir=None,
        set_dir=None,
        image_list=None,
        file_list=None,
        stats_image=None,
        results_image=None,
        battle_message_image=None,
        category=None,
        game_map=None,
        dt=None,
        remove_srcdir=False
):
    if r is None:
        r = ResultSet(
           src_dir=None,
           archive_dir=None,
           error_dir=None,
           set_dir=None,
           image_list=None,
           file_list=None,
           stats_image=None,
           results_image=None,
           battle_message_image=None,
           category=None,
           game_map=None,
           dt=None,
           remove_srcdir=False)
    return ResultSet(
        src_dir = src_dir or r.src_dir,
        archive_dir = archive_dir or r.archive_dir,
        error_dir = error_dir or r.error_dir,
        set_dir = set_dir or r.set_dir,
        image_list = image_list or r.image_list,
        file_list = file_list or r.file_list,
        stats_image = stats_image or r.stats_image,
        results_image = results_image or r.results_image,
        battle_message_image = battle_message_image or r.battle_message_image,
        category = category or r.category,
        game_map = game_map or r.game_map,
        dt = dt or r.dt,
        remove_srcdir = remove_srcdir or r.remove_srcdir)


def base_image(path, dualscreen=IS_DUALSCREEN):
    """Open an image and crop out extra screens

    Postional Arguments:
    path -- path to the image to open, must exist

    Keyword Arguments:
    dualscreen -- boolean should we crop the image to half before working with it

    Returns:
    a PIL image

    Error:
    Doesn't check anything, will raise errors from non-existant files, etc.
    """
    im = Image.open(path)
    if not dualscreen:
        return im

    return im.crop((0, 0, im.width/2, im.height))

def header_image(img, category):
    """Returns the header portion of an image

    Postional Arguments:
    img -- fullscreen PIL image of either the "my results" screen or the "statisics" screen
    category -- GAMECATORY. If naval extra steps are needed

    Returns:
    A PIL image of just the header section of the image
    Error:
    Doesn't check anything, will raise errors on incorrect inputs
    """

    # header region tuple
    # TODO make the lower y bound (box[3]) a ratio as well.
    box = (img.width/3, 0, img.width-(img.width/3), 170)

    cropped_image = img.crop(box)
    # This is dumb. I don't know why enumerations are failing
    if category is GAMECATEGORY.naval:
        cropped_image = mask_naval_symbol(cropped_image)

    return cropped_image

def header_image_to_text(img):
    """Returns the header text from a given (header) img

    Positional Arguments:
    img -- PIL image of just the header section of the "My results" or "statistics" screenshots

    Returns:
    A raw tesseract string

    Error:
    Does check anything, will return the wrong thing or raise an error if the tesseract goes wrong
    """
    return pytesseract.image_to_string(img)

def game_mode(s):
    """Take header string and Return AB||RB||SB

    Positional Arguments:
    s -- A header string, should look like 'Mission Failed\n\nArcade Battles, [Domination] Vietnam\n\x0c'

    Returns:
    A string 'AB' or 'RB' or 'SB'

    Error:
    Raises an error if none of the expected game modes are found in the input string
    """
    if 'Arcade Battles' in s:
        return 'AB'
    elif 'Realistic Battles' in s:
        return 'RB'
    elif 'Simulator Battles' in s:
        return 'SB'
    else:
        raise ValueError('Could not find an expected battle type in the string {}'.format(s))

def map_type(s):
    """Take header string and return a map type

    Positional Arguments:
    s -- A header string, should look like 'Mission Failed\n\nArcade Battles, [Domination] Vietnam\n\x0c'

    Returns:
    A string of the map type (for example, Domination or Conquest #1)

    Error:
    Does not catch regex errors if the regex isn't matched
    """
    return re.search('\[([^\]]*)\]', s).group(1).strip()

def map_name(s):
    """Take header string and return a map name

    Positional Arguments:
    s -- A header string, should look like 'Mission Failed\n\nArcade Battles, [Domination] Vietnam\n\x0c'

    Returns:
    A string of the map name (for example, Vietname)

    Error:
    Does not catch regex errors if the regex isn't matched
    """
    return re.search('\] ([^\n]*)\n', s).group(1).strip()

def w_or_l(s):
    """Take header string and return win or lose indicator


    Positional Arguments:
    s -- A header string, should look like 'Mission Failed\n\nArcade Battles, [Domination] Vietnam\n\x0c'

    Returns:
    A string 'W' or 'L'

    Error:
    Raises an error if none of the expected win or lose strings are found in the input
    """
    if 'Failed' in s:
        return 'L'
    elif 'Accomplished' in s:
        return 'W'
    else:
        raise ValueError("Could not find an victory indication in the given string")

def header_img_from_path(path, dualscreen=IS_DUALSCREEN):
    """Open an image and return the header portion

    Postional Arguments:
    path -- path to the image to open, must exist

    Keyword Arguments:
    dualscreen -- boolean should we crop the image to half before working with it

    Returns:
    a PIL image of the header portion of the WT after-action screen

    Error:
    Doesn't check anything, will raise errors from non-existant files, etc.
    """
    return header_image(base_image(path))

def get_naval_mode_template(path=TEMPLATE_PATHS['naval_mode']):
    """Get the opencv template for the naval mode (header) image

    Keyword Arguments:
    path -- path to the template for the naval mode that shows in the header

    Returns:
    An open CV image of the naval symbol that shows up in naval match headers

    Error:
    Doesn't check anything. Will raise errors on non-existent files, etc.
    """
    return cv2.imread(path, 0)

def mask_naval_symbol(img):
    """Return the coordinates of the naval image if it exists, or (-1, -1, -1, -1)

    Positional arguments:
    img -- the battle result header PIL image

    Returns:
    A set of coordinates defining the matched location, or a set of -1s if no match was found

    Error:
    Does minimal error checking. Will catch if no matches and instead return -1s. Otherwise will raise the source error
    """
    template = get_naval_mode_template()
    w, h = template.shape[::-1]
    imcv = convert_pil_to_cv2(img)

    result = cv2.matchTemplate(imcv,template,cv2.TM_CCOEFF)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < TEMPLATE_THRESHOLDS['naval_mode']:
        print(min_val, max_val, min_loc, max_loc)
        raise ValueError('Could not find naval template in image')
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # Not sure if one color or the other is better here, but so far all is working
    #color = int(imcv[0][0])
    # use the average color of the header to try and approximate the background
    color = int(imcv.mean())

    new_cv_img = cv2.rectangle(imcv,top_left, bottom_right, color, -1)

    return convert_cv2_to_pil(new_cv_img)

def template_match(pil_img, template_name, debug=False):
    """Return True if a match was found

    Positional Arguments:
    pil_img -- A PIL image to search for the given template
    template_path -- path to the template to check

    Returns:
    boolean -- True if the image contains the template

    Error:
    Does not handle errors
    """
    template_path = TEMPLATE_PATHS[template_name]
    template = cv2.imread(template_path,0)
    imcv = np.asarray(pil_img.convert('L'))
    result = cv2.matchTemplate(imcv,template,cv2.TM_CCOEFF)

    if debug:
        return cv2.minMaxLoc(result)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val > TEMPLATE_THRESHOLDS[template_name]:
        return True

    return False

def debug_display_template(template_name):
    """import pil_img and call show

    Positional Arguments:
    template_path -- path for the template to show
    """
    template_path = TEMPLATE_PATHS[template_name]
    im = Image.open(template_path)
    im.show()
    input('press enter for next')


def debug_display_template_match_box(img, template_name):
    """Intended for use with jupyter. Display the header with a box around the naval indication image

    Positional Arguments:
    header_img -- battle result header image in PIL format

    Returns:
    n/a

    Error:
    will just blow up
    """
    min_val, max_val, min_loc, max_loc = template_match(img, template_name, debug=True)
    template = cv2.imread(TEMPLATE_PATHS[template_name],0)
    w, h = template.shape[::-1]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img = convert_pil_to_cv2(img)
    img = cv2.rectangle(img,top_left, bottom_right, 255, 2)
    pil_img = convert_cv2_to_pil(img)
    print(max_val)
    pil_img.show()
    input('press enter for next')

def convert_cv2_to_pil(cv_img):
    """Convert a cv2 image back to pil, to be human visible

    Positional arguments:
    cv_img -- a cv2 image or numpy array? I guess?

    Error:
    Does not do any checking. Will raise any encountered errors
    """
    return Image.fromarray(cv_img)

def convert_pil_to_cv2(pil_img):
    """Convert a PIL image to cv2

    Positional Arguments:
    pil_img -- a PIL image to covnert to cv2 (grayscale)

    Returns:
    the cv2 image

    Errors:
    pass through
    """
    return np.asarray(pil_img.copy().convert('L'))

def find_stats_screen(image_list, category):
    """Identify the statistic page image from a list of PIL images

    Positional Arguments:
    image_list -- a list of PIL images. Should be 3 of them.
    category -- string describing whch category of game (air, ground, naval)

    Returns:
    A single PIL image representing the statistic page

    Exceptions:
    Will raise a value error if a statistics page cannot be found
    """
    template_name = "{}_stats".format(category.name)
    for image in image_list:
        #debug_display_template(template_name)
        #debug_display_template_match_box(image, template_name)
        if template_match(image, template_name):
            return image

    raise ValueError('Could not find a match for the given template in the images')

# TODO merge this with the other find, it's the same method
def find_results_screen(image_list):
    """Identify and return the results image from a list of images

    Positional Arguments:
    image_list -- a list of PIL images, probably ~3 of them

    Returns:
    A single PIL image representing the results page

    Exceptions:
    Will raise a value error if it doesn't find the results image
    """
    for image in image_list:
        if template_match(image, 'results_badge'):
            return image

    raise ValueError('Could not find a match for the given template in the images')

def find_battle_message_screen(image_list):
    """Identify and return the battle message image from a list of images

    Positional Arguments:
    image_list -- a list of PIL images, probably ~3 of them

    Returns:
    A single PIL image representing the battle_message page

    Exceptions:
    Will raise a value error if it doesn't find the battle message image
    """
    template_names = ['battle_message_won_badge', 'battle_message_lost_badge']
    for image in image_list:
        for template_name in template_names:
          if template_match(image, template_name):
              return image

    raise ValueError('Could not find a match for the given template in the images')

def determine_category(image_list):
    """Check for the stats screen from each category. Return the category that matches

    Positional Arguments:
    image_list -- the list of PIL images

    Returns:
    GAMECATEGORY of the matching image

    Exceptions:
    Will raise a value error if none of the images match
    """
    # TODO this is inefficient and will lead to many duplicated calls.
    # Later perhaps we can create a struct to hold values so we do not repeat
    for cat in GAMECATEGORY:
        try:
            find_stats_screen(image_list, cat)
            return cat
        except ValueError:
            pass

    raise ValueError("Could not determine game category from the given images")


# TODO refactor. merge with EventHander._ready
# TODO refactor to use Result (Ok, Err)
def validate_input(directory):
    """Return a list with 3 base images

    Positional Arguments:
    directory -- a directory to check for input files

    Returns:
    list -- a list of 3 PIL images

    Error:
    will raise a ValueError if the input isn't a directory, if there are not 3 files, or if PIL raises an error
    """
    try:
        files = os.listdir(directory)
    except:
        raise ValueError('Could not get a list of files from {}'.format(directory))

    # more checks could go here, but wait to do them until we actually have issues
    # TODO this should not be exactly 3 anymore
    if not len(files) == 3:
        raise ValueError('There was an incorrect number of files in directory {}'.format(directory))

    images = []

    try:
        for file in files:
            images.append(base_image(os.path.join(directory, file)))
    except:
        raise ValueError('Could not process the files in {}. Are they images?'.format(directory))

    return images

def handle_ready_files(ready_dir):
    """Process files in a changed directory

    Positional Arugments:
    ready_dir -- a (validated) ReadyDir with a list of files to process

    Error:
    Should only log errors, never raise them
    """
    worked = utils.railroad(
        init_result_set,
        get_images_list,
        get_result_set,
        archive_set,
        init_arg=ready_dir)

    if worked.is_ok():
        print('done')
        return

    else:
        print('Something when wrong: {}'.format(worked.value))
        archive_error(ready_dir, worked)
        return

# TODO I just rewrote this down in get_error_dirs, so they need to be unified
def init_result_set(ready_dir):
    """Convert the ReadyDir into a partially populated ResultSet

    Positional Arguments:
    ready_dir -- a (validated) ReadyDir with a list of files to process

    Keyword Arguments:
    remove_srcdir -- whether the srcdir should be removed on archive

    Returns:
    Ok or Err -- an Ok wrapping ResultSet with src_dir,archive_ir,error_dir,file_list,dt set, or Err

    Errors:
    Anything unexpected will raise
    """
    r = None
    try:
        first_file = os.path.join(ready_dir.directory, ready_dir.files[0])
        print(ready_dir)
        if ready_dir.play_dt is None:
            set_dt = dt.fromtimestamp(os.stat(first_file).st_mtime).strftime(DT_FORMAT_STR)
        else:
            set_dt = ready_dir.play_dt
        r = update_result(
            src_dir=ready_dir.directory,
            archive_dir=ready_dir.archive_dir,
            error_dir=ready_dir.error_dir,
            file_list=ready_dir.files,
            dt=set_dt,
            remove_srcdir=ready_dir.remove_srcdir)
    except Exception as e:
        return Err(e)

    if r is not None:
        return Ok(r)
    return Err('Something went wrong and we did not create a ResultSet in init_result_set')

def retryable_get_base_image(path, retry_count=3, sleep_base=2):
    """Try to load an image from a file several times, retying if it seems the image is not ready

    Positional Arguments
    path -- path to the file to try to open

    Keyword Arguments
    retry_count -- how many times to retry, default=3
    sleep_base -- how many seconds to sleep after the first attempt, doubled each time. Default 2

    Returns:
    PIL Image loaded from the path

    Errors:
    Anything that we do not expect
    """
    count = 0
    image = None
    sleep = sleep_base
    while count < retry_count:
        try:
            image = base_image(path)
        except OSError as e:
            err_msg = str(e)
            if "image file is truncated" not in err_msg:
                raise e
        if image is not None:
            return image
        time.sleep(sleep)
        count += 1
        sleep *= 2

def get_images_list(r):
    """Return a list of images from all the files in the ready directory

    Positional Arguments:
    r -- A ResultSet within

    Returns:
    Result -- An Ok wrapper for a ResultSet includng the image list, or an Err

    Error:
    Return an Err for error processing
    """
    images = []
    non_image_files = []
    for file in r.file_list:
        try:
            images.append(retryable_get_base_image(os.path.join(r.src_dir, file)))
        except UnidentifiedImageError:
            non_image_files.append(file)

    if len(images) >= 3:
        r = update_result(r=r, image_list=images)
        return Ok(r)

    return Err(non_image_files)

def get_result_set(r):
    """Given a result set, update it with the 3 image types

    Positional Arguments:
    r -- a ResultSet

    Returns
    A result set with stats_image, battle_message_image, and results_image set, or an Err if there are errors

    Errors:
    handled Errors will lead to an Err
    """
    print("Finding stats, results, and battles messages...")
    try:
        category = determine_category(r.image_list)
        stats_image = find_stats_screen(r.image_list, category)
        results_image = find_results_screen(r.image_list)
        battle_message_image = find_battle_message_screen(r.image_list)
        header_img = header_image(stats_image, category)
        header_str = header_image_to_text(header_img)
        r = update_result(
            r=r,
            stats_image=stats_image,
            results_image=results_image,
            battle_message_image=battle_message_image,
            category=category,
            game_map=map_name(header_str))
        print('Done!')
        return Ok(r)
    except Exception as e:
        print('Failed to get the image results: {}'.format(e))
        return Err(e)

# TODO also rename files to match their type
# will require we store  the <type>_image as a struct of its own
def archive_set(result_set):
    """Move the files in the result set from their directory to a new subfolder in archive_dir

    Postiional Arguments:
    result_set -- ResultSet that MUST have set_dir, archive_dir, category, game_map, and dt actually set

    Error:
    will raise any errors
    """
    set_name = dir_name_normalize('{}-{}-{}'.format(result_set.dt, result_set.category.name, result_set.game_map))
    set_dir = result_set.set_dir or os.path.join(
            result_set.archive_dir,
            set_name)
    print('Archiving results to {}'.format(set_dir))
    try:
        os.makedirs(set_dir)
    except FileExistsError:
        pass
    for file in result_set.file_list:
            src = os.path.join(result_set.src_dir, file)
            dest = os.path.join(set_dir, file)
            print('moving {} to {}'.format(src, dest))
            shutil.move(src, dest)
    if result_set.remove_srcdir:
        print('Removing {}'.format(result_set.src_dir))
        shutil.rmtree(result_set.src_dir)
    return Ok()

def dir_name_normalize(s):
    """Remove special characters and downcase directory name

    Positional Arguments:
    s -- string with the expected directory name

    Returns
    a string with no spaces, special characters, and downcased
    """
    # This will probably get more complicated
    return s.lower().replace(' ', '_')

def archive_error(ready_dir, err):
    """Archives all the files of ready_dir to an error directory

    Positional Arguments:
    ready_dir -- A ReadyDir named tuple
    err -- a result with an error, so we can print

    Errors:
    will raise any errors
    """
    print('archive_error called because {}'.format(err.value))
    first_file = os.path.join(ready_dir.directory, ready_dir.files[0])

    approx_time = dt.fromtimestamp(os.stat(first_file).st_mtime).strftime(DT_FORMAT_STR)
    set_dir = os.path.join(ready_dir.error_dir, approx_time)
    try:
        os.makedirs(set_dir)
    except FileExistsError:
        pass
    for file in ready_dir.files:
        src = os.path.join(ready_dir.directory, file)
        dest = os.path.join(set_dir, file)
        shutil.move(src, dest)
    with open(os.path.join(set_dir, 'why'), 'w') as f:
        f.write(str(err.value))

def retry_errors(archive_dir, error_dir):
    """Retry errored directories and archive any successes

    Positional Arguments:
    archive_dir -- The directory to archive any successful retries to
    error_dir -- The directory error directories are in

    Returns:
    Result with a success or failure message

    Errors:
    ValueError if no error directories are found

    Side Effects:
    Will create archive_dir if it does not exist
    """
    r = get_error_dirs(error_dir, archive_dir)
    if r.is_ok():
        rd_list = r.unwrap()
    else:
        print('Failed to get initial directories: {}'.format(r.err()))
        return
    worked = utils.railroad(
        init_each_dir,
        get_images_from_all,
        get_results_from_all,
        archive_all,
        init_arg=rd_list)
    if worked.is_ok():
        print('Some success in retrying errors!')
    else:
        print('No successes in retrying errors: {}'.format(worked.err()))

# TODO all these "loop for a result set functions are gonna be the same
# unify them
def get_images_from_all(rs_list):
    """Run get_images on all result sets and return any successful ones

    Positional Arguments:
    rs_list -- [ResultSet]

    Returns:
    Result -- Ok([ResultSet]) or Err([errors]) if we have no successes
    """
    new_rs_list = []
    error_list = []
    for rs in rs_list:
        r = get_images_list(rs)
        if r.is_ok():
            new_rs_list.append(r.unwrap())
        else:
            error_list.append(r.err())
    if len(new_rs_list) > 0:
        return Ok(new_rs_list)
    else:
        final_error_list = ["get_images_from_all did not have any successes."] + error_list
        return Err(final_error_list)

def get_results_from_all(rs_list):
    """Run get_results on all result sets and return any successful ones

    Positional Arguments:
    rs_list -- [ResultSet]

    Returns:
    Result -- Ok([ResultSet]) or Err([errors]) if we have no successes
    """
    new_rs_list = []
    error_list = []
    for rs in rs_list:
        r = get_result_set(rs)
        if r.is_ok():
            new_rs_list.append(r.unwrap())
        else:
            error_list.append(r.err())

    if len(new_rs_list) > 0:
        return Ok(new_rs_list)
    else:
        final_error_list = ["get_results_from_all did not have any successes."] + error_list
        return Err(final_error_list)

def archive_all(rs_list):
    """Archive all resultsets

    Positional Arguments:
    rs_list -- [ResultSet]

    Returns:
    Result -- Ok([ResultSet]) or Err([errors]) if we have no successes
    """
    new_rs_list = []
    error_list = []
    for rs in rs_list:
        r = archive_set(rs)
        if r.is_ok():
            new_rs_list.append(r.unwrap())
        else:
            error_list.append(r.err())

    if len(new_rs_list) > 0:
        return Ok(new_rs_list)
    else:
        final_error_list = ["archive_all did not have any successes."] + error_list
        return Err(final_error_list)


def init_each_dir(readydir_list):
    """Take a list of ReadyDirs representing possible ResultSets and return ResultSets

    Positional Arguments:
    readydir_list -- a list of readydirs to convert to resultsets

    Returns:
    Result -- Ok[ResultSet] or Err
    """
    resultset_list = []
    error_list = []
    for rd in readydir_list:
        r = init_result_set(rd)
        if r.is_ok():
            resultset_list.append(r.unwrap())
        else:
            error_list.append(r.err())
    if len(resultset_list) > 0:
        return Ok(resultset_list)

    return Err(error_list)

def get_error_dirs(error_dir, archive_dir):
    """Retrieve a list of subdirectories that meet the error dir criteria

    Error formatted directories are simply a DT_FORMAT_STR datetime

    Positional Arguments:
    error_dir -- string to the directory to look in
    archive_dir -- string path of the directory to archive succes to, used to construct ReadyDirs

    Returns:
    Result -- Ok([wtde.ReadyDir]), Err() if no directories found
    """
    sub_dirs = []
    error_dirs = []
    fail_list = []
    try:
        sub_dirs = os.listdir(error_dir)
    except FileNotFoundError:
        pass

    for sd in sub_dirs:
        d = os.path.join(error_dir, sd)
        play_dt = None
        try:
            play_dt = dt.strptime(sd, DT_FORMAT_STR)
        except ValueError as e:
            if 'does not match format' in str(e) or 'unconverted data remains' in str(e):
                print('Directory {} name not in error format, skipping'.format(sd))
                pass
            else:
                fail_list.append(e)
                pass

        # check to make sure we have at least 3 images in the given directory
        image_file_list = [f for f in os.listdir(d) if imghdr.what(os.path.join(d, f))]
        if len(image_file_list) < 3:
            print('There are less than 3 images in {}. Skipping.'.format(d))
            continue

        # we have enough images in a directory we expected.
        rd = ReadyDir(
            files=image_file_list,
            directory=d,
            archive_dir=archive_dir,
            error_dir=error_dir,
            play_dt=play_dt,
            remove_srcdir=True)

        error_dirs.append(rd)

    if len(error_dirs) == 0:
        return Err(fail_list)
    return Ok(error_dirs)

