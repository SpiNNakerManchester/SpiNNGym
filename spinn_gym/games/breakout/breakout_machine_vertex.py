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

from enum import Enum

from spinn_utilities.overrides import overrides

from data_specification.enums.data_type import DataType

# PACMAN imports
from pacman.executor.injection_decorator import inject_items

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.abstract_models.\
    abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants

from spynnaker.pyNN.utilities import constants

# spinn_gym imports
from spinn_gym.games import SpinnGymMachineVertex


# ----------------------------------------------------------------------------
# BreakoutMachineVertex
# ----------------------------------------------------------------------------
class BreakoutMachineVertex(SpinnGymMachineVertex):
    BREAKOUT_REGION_BYTES = 4
    PARAM_REGION_BYTES = 40

    _BREAKOUT_REGIONS = Enum(
        value="_BREAKOUT_REGIONS",
        names=[('SYSTEM', 0),
               ('BREAKOUT', 1),
               ('RECORDING', 2),
               ('PARAMS', 3)])

    def __init__(self,  n_neurons, constraints, label,
                 app_vertex, x_factor, y_factor, colour_bits,
                 simulation_duration_ms, bricking,
                 rand_seed):
        # Superclasses
        super(BreakoutMachineVertex, self).__init__(
            label, constraints, app_vertex, n_neurons,
            self.BREAKOUT_REGION_BYTES + self.PARAM_REGION_BYTES,
            simulation_duration_ms, rand_seed)

        self._x_factor = x_factor
        self._y_factor = y_factor
        self._colour_bits = colour_bits

        self._bricking = bricking

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"routing_info": "RoutingInfos"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"routing_info"}
               )
    def generate_data_specification(self, spec, placement, routing_info):
        # pylint: disable=arguments-differ
        vertex = placement.vertex

        spec.comment("\n*** Spec for Breakout Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=BreakoutMachineVertex._BREAKOUT_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=BreakoutMachineVertex._BREAKOUT_REGIONS.BREAKOUT.value,
            size=self.BREAKOUT_REGION_BYTES, label='BreakoutKey')
        # Reserve recording region
        spec.reserve_memory_region(
            BreakoutMachineVertex._BREAKOUT_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=BreakoutMachineVertex._BREAKOUT_REGIONS.PARAMS.value,
            size=self.PARAM_REGION_BYTES, label='Parameters')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            BreakoutMachineVertex._BREAKOUT_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name()))

        # Write breakout region containing routing key to transmit with
        spec.comment("\nWriting breakout region:\n")
        spec.switch_write_focus(
            BreakoutMachineVertex._BREAKOUT_REGIONS.BREAKOUT.value)
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.SPIKE_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting breakout recording region:\n")
        spec.switch_write_focus(
            BreakoutMachineVertex._BREAKOUT_REGIONS.RECORDING.value)
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size]))

        spec.comment("\nWriting breakout param region:\n")
        spec.switch_write_focus(
            BreakoutMachineVertex._BREAKOUT_REGIONS.PARAMS.value)
        spec.write_value(self._x_factor, data_type=DataType.UINT32)
        spec.write_value(self._y_factor, data_type=DataType.UINT32)
        spec.write_value(self._bricking, data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[3], data_type=DataType.UINT32)

        # End-of-Spec:
        spec.end_specification()

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._BREAKOUT_REGIONS.RECORDING.value, txrx)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "breakout.aplx"
