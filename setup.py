"""Setup for pip package."""

import importlib.util
import subprocess
import os
import setuptools
from setuptools import Extension

__contact_emails__ = ''
__contact_names__ = ''
__description__ = 'A Library to Train Hybrid Parallel Deep Neural Models'
__download_url__ = 'https://github.com/tencent/gcore'
__homepage__ = 'https://github.com/tencent/gcore'
__keywords__ = 'deep learning, transformer, torch, reinforce learning, multi modal'
__license__ = 'BSD-3'
__package_name__ = 'gcore'
__repository_url__ = 'https://github.com/tencent/gcore'
__version__ = '4.2.2'

__version__ += '+' + subprocess.getoutput('git rev-parse --short HEAD | cut -c 1-8').strip()

long_description = ''
long_description_content_type = "text/markdown"

install_requires = []

setuptools.setup(
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    # The project's main homepage.
    url=__repository_url__,
    download_url=__download_url__,
    # Author details
    author=__contact_names__,
    author_email=__contact_emails__,
    # maintainer Details
    maintainer=__contact_names__,
    maintainer_email=__contact_emails__,
    # The licence under which the project is released
    license=__license__,
    packages=setuptools.find_namespace_packages(include=[
        'gpatch',
        'gpatch.*',
        'mpatch',
        'mpatch.*',
        'gdataset',
        'gdataset.*',
        'gpatch_v4',
        'gpatch_v4.*',
    ]),
    ext_modules=[],
    # Add in any packaged data.
    include_package_data=True,
    # PyPI package information.
    keywords=__keywords__,
    install_requires=install_requires,
)
