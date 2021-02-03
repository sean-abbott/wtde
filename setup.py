import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="wtde",
    version=read("VERSION").strip(),
    url="https://github.com/sean-abbott/wtde",
    license='MIT',

    author="Sean Abbott",
    author_email="sabbott.official@gmail.com",

    description="extract useful and fun data from screenshots of your war thunder mission reports",

    packages=find_packages('src', exclude=('tests',)),
    package_dir={'': 'src'},

    install_requires=[
        'fire',
        'matplotlib',
        'numpy',
        'opencv-python',
        'pandas',
        'pillow',
        'pytesseract'
    ],

    extras_require = {
        'jupyter': [
            'jupyter'
        ]
#        'test': [
#            'pytest',
#            'pytest-env'
#        ]
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'wtde = wtde.extract_mission:main'
        ]
    },

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
