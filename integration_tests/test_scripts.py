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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from spinnaker_testbase import ScriptChecker
from unittest import SkipTest  # pylint: disable=unused-import


class TestScripts(ScriptChecker):
    """
    This file tests the scripts as configured in script_builder.py

    Please do not manually edit this file.
    It is rebuilt each time SpiNNakerManchester/IntegrationTests is run

    If it is out of date please edit and run script_builder.py
    Then the new file can be added to github for reference only.
    """
# flake8: noqa

    def test_examples_store_recall_store_recall_test(self):
        self.check_script("examples/store_recall/store_recall_test.py")

    def test_examples_double_inverted_pendulum_double_inverted_pendulum_test(self):
        self.check_script("examples/double_inverted_pendulum/double_inverted_pendulum_test.py")

    def test_examples_logic_logic_test(self):
        self.check_script("examples/logic/logic_test.py")

    def test_examples_breakout_breakout_simple_connection(self):
        self.check_script("examples/breakout/breakout_simple_connection.py")

    def test_examples_breakout_breakout_automated(self):
        self.check_script("examples/breakout/automated_bkout_play.py")

    def test_examples_breakout_breakout_neuromodulated(self):
        self.check_script("examples/breakout/neuromodulated_bkout_play.py")

    def test_examples_inverted_pendulum_inverted_pendulum_test(self):
        self.check_script("examples/inverted_pendulum/inverted_pendulum_test.py")

    def test_examples_multi_arm_bandit_bandit_test(self):
        self.check_script("examples/multi_arm_bandit/bandit_test.py")
