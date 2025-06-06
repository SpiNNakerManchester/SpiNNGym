# Copyright (c) 2017 The University of Manchester
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

import os
import unittest
from spinn_utilities.configs.config_checker import ConfigChecker
from spynnaker.pyNN.config_setup import unittest_setup
import spinn_gym


class TestCfgChecker(unittest.TestCase):

    def setUp(self):
        unittest_setup()

    def test_config_checks(self):
        unittests = os.path.dirname(__file__)
        parent = os.path.dirname(unittests)
        spinn_gym_dir = spinn_gym.__path__[0]
        examples = os.path.join(parent, "examples")
        integration_tests = os.path.join(parent, "integration_tests")
        checker = ConfigChecker([spinn_gym_dir, examples, integration_tests, unittests])
        checker.check(local_defaults=False)
