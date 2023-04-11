# Copyright (c) 2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import distutils.dir_util
from setuptools import setup
import os
import sys


if __name__ == '__main__':
    # Repeated installs assume files have not changed
    # https://github.com/pypa/setuptools/issues/3236
    if len(sys.argv) > 0 and sys.argv[1] == 'egg_info':
        # on the first call to setpy.py remove files left by previous install
        this_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(this_dir, "build")
        if os.path.isdir(build_dir):
            distutils.dir_util.remove_tree(build_dir)
        egg_dir = os.path.join(
            this_dir, "SpiNNGym.egg-info")
        if os.path.isdir(egg_dir):
            distutils.dir_util.remove_tree(egg_dir)
    setup()
