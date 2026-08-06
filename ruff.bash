#!/bin/bash
# Copyright (c) 2020 The University of Manchester
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

# This bash assumes that other repositories are installed in parallel
# ruffs SpiNNUtils, spinn_machine and unittests

if [ "$#" -eq  "0" ]
  then
    echo "Using previous setup. Provide an argument to run setup"
    source ../SupportScripts/venv/ruff_runner/bin/activate
else
  python3 -m venv ../SupportScripts/venv/ruff_runner
  source ../SupportScripts/venv/ruff_runner/bin/activate
  python3 -m pip install --upgrade ruff
fi

echo using ruff.toml
ruff check ../SpiNNUtils/spinn_utilities ../SpiNNUtils/unittests \
    ../SpiNNMachine/spinn_machine ../SpiNNMachine/unittests \
    ../SpiNNMan/spinnman ../SpiNNMan/unittests \
    ../SpiNNMan/spinnman_integration_tests ../SpiNNMan/manual_scripts \
    ../PACMAN/pacman ../PACMAN/pacman_test_objects ../PACMAN/unittests \
    ../spalloc/spalloc_client ../spalloc/tests \
     ../SpiNNFrontEndCommon/spinn_front_end_common ../SpiNNFrontEndCommon/unittests \
     ../SpiNNFrontEndCommon/fec_integration_tests \
     ../TestBase/spinnaker_testbase ../TestBase/unittests \
     ../sPyNNaker/spynnaker ../sPyNNaker/unittests \
     ../sPyNNaker/spynnaker_integration_tests ../sPyNNaker/proxy_integration_tests \
     examples spinn_gym integration_tests \
     --target-version py310 --config ../SupportScripts/actions/ruff/ruff.toml
echo using ruff_up.toml
ruff check ../SpiNNUtils/spinn_utilities ../SpiNNUtils/unittests \
    ../SpiNNMachine/spinn_machine ../SpiNNMachine/unittests \
    ../SpiNNMan/spinnman ../SpiNNMan/unittests \
    ../SpiNNMan/spinnman_integration_tests ../SpiNNMan/manual_scripts \
    ../PACMAN/pacman ../PACMAN/pacman_test_objects ../PACMAN/unittests \
    ../spalloc/spalloc_client ../spalloc/tests \
     ../SpiNNFrontEndCommon/spinn_front_end_common ../SpiNNFrontEndCommon/unittests \
     ../SpiNNFrontEndCommon/fec_integration_tests \
     ../TestBase/spinnaker_testbase ../TestBase/unittests \
     ../sPyNNaker/spynnaker ../sPyNNaker/unittests \
     ../sPyNNaker/spynnaker_integration_tests ../sPyNNaker/proxy_integration_tests \
     examples spinn_gym integration_tests \
     --target-version py310 --config ../SupportScripts/actions/ruff/ruff_up.toml