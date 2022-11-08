# Copyright (c) 2019-2022 The University of Manchester
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

import numpy
from enum import Enum

from spinn_utilities.overrides import overrides

from data_specification.enums.data_type import DataType

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.abstract_models \
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants

# sPyNNaker imports
from spynnaker.pyNN.data import SpynnakerDataView
from spynnaker.pyNN.utilities import constants

# spinn_gym imports
from spinn_gym.games import SpinnGymMachineVertex


# ----------------------------------------------------------------------------
# BanditMachineVertex
# ----------------------------------------------------------------------------
class BanditMachineVertex(SpinnGymMachineVertex):
    BANDIT_REGION_BYTES = 4
    BASE_ARMS_REGION_BYTES = 11 * 4

    _BANDIT_REGIONS = Enum(
        value="_BANDIT_REGIONS",
        names=[('SYSTEM', 0),
               ('BANDIT', 1),
               ('RECORDING', 2),
               ('ARMS', 3)])

    __slots__ = ["_arms", "_constant_input", "_no_arms", "_rate_off",
                 "_rate_on", "_reward_based", "_reward_delay", "_stochastic"]

    def __init__(self, label, app_vertex, n_neurons, simulation_duration_ms,
                 random_seed, arms, reward_delay, reward_based, rate_on,
                 rate_off, stochastic, constant_input):
        """

        :param label: The optional name of the vertex
        :type label: str or None
        :param iterable(AbstractConstraint) constraints:
            The optional initial constraints of the vertex
        :param app_vertex:
            The application vertex that caused this machine vertex to be
            created. If None, there is no such application vertex.
        :type app_vertex: ApplicationVertex or None
        :param int n_neurons:
            The number of neurons to be used to create the slice of the
            application vertex that this machine vertex implements.
        :param int region_bytes: The bytes needed other than recording
        :param float simulation_duration_ms:
        :param list(int) random_seed: List of 4 vlaues to seed the c code
        :param random_seed:
        :param arms:
        :param reward_delay:
        :param reward_based:
        :param rate_on:
        :param rate_off:
        :param stochastic:
        :param constant_input:

        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        :raises PacmanValueError: If the slice of the machine_vertex is too big
        :raise AttributeError:
            If a not None app_vertex is not an ApplicationVertex

        """

        # Superclasses
        super(BanditMachineVertex, self).__init__(
            label, app_vertex, n_neurons,
            self.BANDIT_REGION_BYTES + self.BASE_ARMS_REGION_BYTES,
            simulation_duration_ms,  random_seed)

        # Pass in variables
        arms_list = []
        for arm in arms:
            arms_list.append(numpy.uint32(arm*0xffffffff))
        self._arms = arms_list

        self._no_arms = len(arms)

        self._reward_delay = reward_delay
        self._reward_based = reward_based

        self._rate_on = rate_on
        self._rate_off = rate_off
        self._stochastic = stochastic
        self._constant_input = constant_input

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
    def generate_data_specification(self, spec, placement):
        # pylint: disable=arguments-differ
        vertex = placement.vertex

        spec.comment("\n*** Spec for Bandit Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=self._BANDIT_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=self._BANDIT_REGIONS.BANDIT.value,
            size=self.BANDIT_REGION_BYTES, label='BanditParams')
        # reserve recording region
        spec.reserve_memory_region(
            self._BANDIT_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=self._BANDIT_REGIONS.ARMS.value,
            size=self.BASE_ARMS_REGION_BYTES+(self._no_arms*4),
            label='BanditArms')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            self._BANDIT_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name()))

        # Write bandit region containing routing key to transmit with
        spec.comment("\nWriting bandit region:\n")
        spec.switch_write_focus(
            self._BANDIT_REGIONS.BANDIT.value)
        routing_info = SpynnakerDataView.get_routing_infos()
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.SPIKE_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting bandit recording region:\n")
        spec.switch_write_focus(
            self._BANDIT_REGIONS.RECORDING.value)
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size]))

        # Write probabilites for arms
        spec.comment("\nWriting arm probability region:\n")
        spec.switch_write_focus(
            self._BANDIT_REGIONS.ARMS.value)
        spec.write_value(self._reward_delay, data_type=DataType.UINT32)
        spec.write_value(self._no_arms, data_type=DataType.UINT32)
        spec.write_value(self._random_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[3], data_type=DataType.UINT32)
        spec.write_value(self._reward_based, data_type=DataType.UINT32)
        spec.write_value(self._rate_on, data_type=DataType.UINT32)
        spec.write_value(self._rate_off, data_type=DataType.UINT32)
        spec.write_value(self._stochastic, data_type=DataType.UINT32)
        spec.write_value(self._constant_input, data_type=DataType.UINT32)
        # Write the data - Arrays must be 32-bit values, so convert
        data = numpy.array(self._arms, dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))

        # End-of-Spec:
        spec.end_specification()

    def get_recording_region_base_address(self, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._BANDIT_REGIONS.RECORDING.value)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "bandit.aplx"
