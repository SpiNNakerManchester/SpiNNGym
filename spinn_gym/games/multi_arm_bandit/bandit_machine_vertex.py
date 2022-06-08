# Copyright (c) 2019-2021 The University of Manchester
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

# PACMAN imports
from pacman.executor.injection_decorator import inject_items

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

    def __init__(self, n_neurons, sdram_required, constraints, label,
                 app_vertex, arms, reward_delay, reward_based, rate_on,
                 rate_off, stochastic, constant_input,
                 simulation_duration_ms, rand_seed):

        # Superclasses
        super(BanditMachineVertex, self).__init__(
            label, constraints, app_vertex, n_neurons, sdram_required,
            simulation_duration_ms,  rand_seed)

        # Pass in variables
        arms_list = []
        for arm in arms:
            arms_list.append(numpy.uint32(arm*0xffffffff))
        self._arms = arms_list

        self._no_arms = len(arms)
        self._rand_seed = rand_seed

        self._reward_delay = reward_delay
        self._reward_based = reward_based

        self._rate_on = rate_on
        self._rate_off = rate_off
        self._stochastic = stochastic
        self._constant_input = constant_input

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"routing_info": "RoutingInfos"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"routing_info"}
               )
    def generate_data_specification(self, spec, placement, routing_info):
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
        spec.write_value(self._rand_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[3], data_type=DataType.UINT32)
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

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._BANDIT_REGIONS.RECORDING.value, txrx)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "bandit.aplx"
