import os
import re

from enum import Enum

import cv2
import numpy as np
import pytesseract

from matplotlib import pyplot as plt
from PIL import Image

IS_DUALSCREEN = True

TEMPLATE_PATHS = {
    'naval_mode': 'template_images/NavalMatch.png'
}

# I will likely have to play with this.
TEMPLATE_MATCH_THRESHOLD = 60000000

TEMPLATE_DIR = 'template_images'

GAMECATEGORY = Enum('GAMECATEGORY', 'ground air naval')

STATS_TEMPLATES = {
    GAMECATEGORY.ground: os.path.join(TEMPLATE_DIR, 'ground_stats.png'),
    GAMECATEGORY.air: os.path.join(TEMPLATE_DIR, 'air_stats.png'),
    GAMECATEGORY.naval: os.path.join(TEMPLATE_DIR, 'naval_stats.png'),
}


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

def header_image(img):
    """Returns the header portion of an image

    Postional Arguments:
    img -- fullscreen PIL image of either the "my results" screen or the "statisics" screen

    Returns:
    A PIL image of just the header section of the image
Error:
    Doesn't check anything, will raise errors on incorrect inputs
    """

    # header region tuple
    # TODO make the lower y bound (box[3]) a ratio as well.
    box = (img.width/3, 0, img.width-(img.width/3), 170)

    return img.crop(box)

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
    return re.search('\[([^\]]*)\]', s).group(1)

def map_name(s):
    """Take header string and return a map name

    Positional Arguments:
    s -- A header string, should look like 'Mission Failed\n\nArcade Battles, [Domination] Vietnam\n\x0c'

    Returns:
    A string of the map name (for example, Vietname)

    Error:
    Does not catch regex errors if the regex isn't matched
    """
    return re.search('\] ([^\n]*)\n', s).group(1)

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
    elif 'Success' in s:
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

def header_is_naval_by_template(img):
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
    imcv = np.asarray(img.convert('L'))

    try:
        result = cv2.matchTemplate(imcv,template,cv2.TM_CCOEFF)
    # TODO actually catch the failing error or check for failing condition
    except Error as e:
        print('If this was a "not found" error, we should instead return ((-1, -1), (-1, -1))')
        raise e

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return (top_left, bottom_right)

def template_match(pil_img, template_path):
    """Return True if a match was found

    Positional Arguments:
    pil_img -- A PIL image to search for the given template
    template_path -- path to the template to check

    Returns:
    boolean -- True if the image contains the template

    Error:
    Does not handle errors
    """
    template = cv2.imread(template_path,0)
    imcv = np.asarray(pil_img.convert('L'))
    result = cv2.matchTemplate(imcv,template,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val > TEMPLATE_MATCH_THRESHOLD:
        return True

    return False

def debug_display_template(template_path):
    """import pil_img and call show

    Positional Arguments:
    template_path -- path for the template to show
    """
    im = Image.open(template_path)
    im.show()


def debug_display_template_match_box(img, template_path):
    """Intended for use with jupyter. Display the header with a box around the naval indication image

    Positional Arguments:
    header_img -- battle result header image in PIL format

    Returns:
    n/a

    Error:
    will just blow up
    """
    min_val, max_val, min_loc, max_loc = template_match(img, template_path)
    template = cv2.imread(template_path,0)
    w, h = template.shape[::-1]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img = convert_pil_to_cv2(img)
    img = cv2.rectangle(img,top_left, bottom_right, 255, 2)
    pil_img = convert_cv2_to_pil(img)
    pil_img.show()

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
    category -- GAMECATEGORY describing whch category of game (air, ground, naval)

    Returns:
    A single PIL image representing the statistic page

    Exceptions:
    Will raise a value error if a statistics page cannot be found
    """
    # TODO finish this
    template = STATS_TEMPLATES[GAMECATEGORY[category]]
    for image in image_list:
        #print(template_match(image, template))
        #debug_display_template_match_box(image, template)
        #input('press enter for next')
        if template_match(image, template):
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
    # TODO this is inefficient and will lead to many duplicated matches.
    # Later perhaps we can create a struct to hold values so we do not repeat
    for cat in GAMECATEGORY:
        try:
            find_stats_screen(image_list, cat.name)
            return cat
        except ValueError:
            pass


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
    if not len(files) == 3:
        raise ValueError('There was an incorrect number of files in directory {}'.format(directory))

    images = []

    try:
        for file in files:
            images.append(base_image(os.path.join(directory, file)))
    except:
        raise ValueError('Could not process the files in {}. Are they images?'.format(directory))

    return images
