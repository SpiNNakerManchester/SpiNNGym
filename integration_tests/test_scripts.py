# Copyright (c) 2019 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from spinnaker_testbase import ScriptChecker


class TestScripts(ScriptChecker):
    """
    This file tests the scripts as configured in script_builder.py

    Please do not manually edit this file.
    It is rebuilt each time SpiNNakerManchester/IntegrationTests is run

    If it is out of date please edit and run script_builder.py
    Then the new file can be added to github for reference only.
    """
# flake8: noqa

    def test_examples_double_inverted_pendulum_double_inverted_pendulum_test(self):
        self.check_script("examples/double_inverted_pendulum/double_inverted_pendulum_test.py")

    def test_examples_logic_logic_test(self):
        self.check_script("examples/logic/logic_test.py")

    def test_examples_breakout_breakout_simple_connection(self):
        self.check_script("examples/breakout/breakout_simple_connection.py")

    # Not testing file due to: Runs forever
    # examples/breakout/automated.py

    def test_examples_breakout_neuromodulated_bkout_play(self):
        self.check_script("examples/breakout/neuromodulated_bkout_play.py")

    def test_examples_breakout_automated_bkout_play(self):
        self.check_script("examples/breakout/automated_bkout_play.py")

    def test_examples_store_recall_store_recall_test(self):
        self.check_script("examples/store_recall/store_recall_test.py")

    # Not testing file due to: Not a script
    # examples/icub_vor_env/icub_utilities.py

    def test_examples_icub_vor_env_icub_vor_env_test_200_inputs(self):
        self.check_script("examples/icub_vor_env/icub_vor_env_test_200_inputs.py")

    def test_examples_icub_vor_env_icub_vor_venv_test_perfect_motion(self):
        self.check_script("examples/icub_vor_env/icub_vor_venv_test_perfect_motion.py")

    def test_examples_icub_vor_env_icub_vor_env_test(self):
        self.check_script("examples/icub_vor_env/icub_vor_env_test.py")

    def test_examples_multi_arm_bandit_bandit_test(self):
        self.check_script("examples/multi_arm_bandit/bandit_test.py")

    def test_examples_inverted_pendulum_inverted_pendulum_test(self):
        self.check_script("examples/inverted_pendulum/inverted_pendulum_test.py")
