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

from enum import Enum
import numpy

from spinn_utilities.overrides import overrides

from pacman.model.placements import Placement

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.abstract_models \
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.interface.ds import (
    DataSpecificationGenerator, DataType)
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants

# sPyNNaker imports
from spynnaker.pyNN.data import SpynnakerDataView
# spinn_gym imports
from spinn_gym.games import SpinnGymMachineVertex


# ----------------------------------------------------------------------------
# LogicMachineVertex
# ----------------------------------------------------------------------------
class LogicMachineVertex(SpinnGymMachineVertex):
    LOGIC_REGION_BYTES = 4
    BASE_DATA_REGION_BYTES = 9 * 4

    _LOGIC_REGIONS = Enum(
        value="_LOGIC_REGIONS",
        names=[('SYSTEM', 0),
               ('LOGIC', 1),
               ('RECORDING', 2),
               ('DATA', 3)])

    __slots__ = ("_input_sequence", "_no_inputs", "_rate_on", "_rate_off",
                 "_score_delay", "_stochastic", "_truth_table")

    def __init__(self, label, app_vertex, n_neurons,
                 simulation_duration_ms, random_seed,
                 truth_table, input_sequence, rate_on, rate_off,
                 score_delay, stochastic):
        """

        :param label: The optional name of the vertex
        :type label: str or None
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
        :param truth_table:
        :param input_sequence:
        :param rate_on:
        :param rate_off:
        :param score_delay:
        :param stochastic:

        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        :raises PacmanValueError: If the slice of the machine_vertex is too big
        :raise AttributeError:
            If a not None app_vertex is not an ApplicationVertex
        """

        # Superclasses
        super(LogicMachineVertex, self).__init__(
            label, app_vertex, n_neurons,
            self.LOGIC_REGION_BYTES + self.BASE_DATA_REGION_BYTES,
            simulation_duration_ms,  random_seed)

        # Pass in variables
        self._truth_table = truth_table
        self._rate_on = rate_on
        self._rate_off = rate_off
        self._stochastic = stochastic
        self._input_sequence = input_sequence
        self._no_inputs = len(input_sequence)
        self._score_delay = score_delay

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
    def generate_data_specification(
            self, spec: DataSpecificationGenerator, placement: Placement):
        # pylint: disable=arguments-differ
        vertex = placement.vertex

        spec.comment("\n*** Spec for Logic Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=self._LOGIC_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=self._LOGIC_REGIONS.LOGIC.value,
            size=self.LOGIC_REGION_BYTES, label='LogicParams')
        # reserve recording region
        spec.reserve_memory_region(
            self._LOGIC_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=self._LOGIC_REGIONS.DATA.value,
            size=self.BASE_DATA_REGION_BYTES+(self._no_inputs*4)+(
                len(self._truth_table)*4),
            label='LogicArms')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            self._LOGIC_REGIONS.SYSTEM.value)
        assert isinstance(vertex, AbstractHasAssociatedBinary)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name()))

        # Write logic region containing routing key to transmit with
        spec.comment("\nWriting logic region:\n")
        spec.switch_write_focus(
            self._LOGIC_REGIONS.LOGIC.value)
        routing_info = SpynnakerDataView.get_routing_infos()
        spec.write_value(routing_info.get_single_first_key_from_pre_vertex(
            vertex))

        # Write recording region for score
        spec.comment("\nWriting logic recording region:\n")
        spec.switch_write_focus(
            self._LOGIC_REGIONS.RECORDING.value)
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size]))

        # Write logic data
        spec.comment("\nWriting logic data region:\n")
        spec.switch_write_focus(
            self._LOGIC_REGIONS.DATA.value)
        spec.write_value(self._score_delay, data_type=DataType.UINT32)
        spec.write_value(self._no_inputs, data_type=DataType.UINT32)
        spec.write_value(self._random_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[3], data_type=DataType.UINT32)
        spec.write_value(self._rate_on, data_type=DataType.UINT32)
        spec.write_value(self._rate_off, data_type=DataType.UINT32)
        spec.write_value(self._stochastic, data_type=DataType.UINT32)
        # Write the data - Arrays must be 32-bit values, so convert
        data = numpy.array(self._input_sequence, dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))
        data = numpy.array(self._truth_table, dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))

        # End-of-Spec:
        spec.end_specification()

    def get_recording_region_base_address(self, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._LOGIC_REGIONS.RECORDING.value)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self) -> str:
        return "logic.aplx"
