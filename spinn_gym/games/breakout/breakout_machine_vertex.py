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

from enum import Enum

from spinn_utilities.overrides import overrides

from data_specification.enums.data_type import DataType

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

from spynnaker.pyNN.data import SpynnakerDataView
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

    __slots__ = ["_x_factor", "_y_factor", "_colour_bits", "_bricking"]

    def __init__(
            self, label, constraints, app_vertex, n_neurons,
            simulation_duration_ms, random_seed,
            x_factor, y_factor, colour_bits, bricking):
        """

        :param label: The optional name of the vertex
        :type label: str or None
        :param iterable(AbstractConstraint) constraints:
            The optional initial constraints of the vertex
        :type constraints: iterable(AbstractConstraint) or None
        :type constraints: iterable(AbstractConstraint)  or None
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
        :param x_factor:
        :param y_factor:
        :param colour_bits:
        :param bricking:

        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        :raises PacmanValueError: If the slice of the machine_vertex is too big
        :raise AttributeError:
            If a not None app_vertex is not an ApplicationVertex
        """
        # Superclasses
        super(BreakoutMachineVertex, self).__init__(
            label, constraints, app_vertex, n_neurons,
            self.BREAKOUT_REGION_BYTES + self.PARAM_REGION_BYTES,
            simulation_duration_ms, random_seed)

        self._x_factor = x_factor
        self._y_factor = y_factor
        self._colour_bits = colour_bits
        self._bricking = bricking

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
    def generate_data_specification(self, spec, placement):
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
        routing_info = SpynnakerDataView.get_routing_infos()
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
        spec.write_value(self._random_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[3], data_type=DataType.UINT32)

        # End-of-Spec:
        spec.end_specification()

    @overrides(SpinnGymMachineVertex.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, _partition):
        return self._vertex_slice.n_atoms

    @overrides(SpinnGymMachineVertex.get_recording_region_base_address)
    def get_recording_region_base_address(self, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._BREAKOUT_REGIONS.RECORDING.value)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "breakout.aplx"
