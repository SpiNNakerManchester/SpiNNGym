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
from spynnaker.pyNN.utilities.constants import SPIKE_PARTITION_ID

# spinn_gym imports
from spinn_gym.games import SpinnGymMachineVertex


# ----------------------------------------------------------------------------
# RecallMachineVertex
# ----------------------------------------------------------------------------
class RecallMachineVertex(SpinnGymMachineVertex):
    RECALL_REGION_BYTES = 4
    DATA_REGION_BYTES = 12 * 4

    _RECALL_REGIONS = Enum(
        value="_RECALL_REGIONS",
        names=[('SYSTEM', 0),
               ('RECALL', 1),
               ('RECORDING', 2),
               ('DATA', 3)])

    __slots__ = ("_prob_command", "_prob_in_change", "_pop_size",
                 "_rate_off", "_rate_on", "_reward", "_stochastic",
                 "_time_period")

    def __init__(self, label,  app_vertex, n_neurons,
                 simulation_duration_ms, random_seed,
                 rate_on, rate_off, pop_size, prob_command,
                 prob_in_change, time_period, stochastic, reward):
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
        :param rate_on:
        :param rate_off:
        :param pop_size:
        :param prob_command:
        :param prob_in_change:
        :param time_period:
        :param stochastic:
        :param reward:

        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        :raises PacmanValueError: If the slice of the machine_vertex is too big
        :raise AttributeError:
            If a not None app_vertex is not an ApplicationVertex

        """

        # Superclasses
        super(RecallMachineVertex, self).__init__(
            label, app_vertex, n_neurons,
            self.RECALL_REGION_BYTES + self.DATA_REGION_BYTES,
            simulation_duration_ms,  random_seed)
        # Pass in variables
        self._rate_on = rate_on
        self._rate_off = rate_off
        self._stochastic = stochastic
        self._reward = reward
        self._pop_size = pop_size
        self._prob_command = prob_command
        self._prob_in_change = prob_in_change
        self._time_period = time_period

        # used to define size of recording region
        self._recording_size = int((simulation_duration_ms / 1000.) * 4)

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
    def generate_data_specification(self, spec: DataSpecificationGenerator,
                                    placement: Placement) -> None:
        # pylint: disable=arguments-differ
        vertex = placement.vertex

        spec.comment("\n*** Spec for Recall Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=self._RECALL_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=self._RECALL_REGIONS.RECALL.value,
            size=self.RECALL_REGION_BYTES, label='RecallParams')
        # reserve recording region
        spec.reserve_memory_region(
            self._RECALL_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=self._RECALL_REGIONS.DATA.value,
            size=self.DATA_REGION_BYTES, label='RecallArms')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            self._RECALL_REGIONS.SYSTEM.value)
        assert isinstance(vertex, AbstractHasAssociatedBinary)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name()))

        # Write recall region containing routing key to transmit with
        spec.comment("\nWriting recall region:\n")
        spec.switch_write_focus(
            self._RECALL_REGIONS.RECALL.value)
        routing_info = SpynnakerDataView.get_routing_infos()
        spec.write_value(routing_info.get_key_from(
            vertex, SPIKE_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting recall recording region:\n")
        spec.switch_write_focus(
            self._RECALL_REGIONS.RECORDING.value)
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size]))

        # Write probabilites for arms
        spec.comment("\nWriting recall data region:\n")
        spec.switch_write_focus(
            self._RECALL_REGIONS.DATA.value)
        spec.write_value(self._time_period, data_type=DataType.UINT32)
        spec.write_value(self._pop_size, data_type=DataType.UINT32)
        spec.write_value(self._random_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[3], data_type=DataType.UINT32)
        spec.write_value(self._rate_on, data_type=DataType.UINT32)
        spec.write_value(self._rate_off, data_type=DataType.UINT32)
        spec.write_value(self._stochastic, data_type=DataType.UINT32)
        spec.write_value(self._reward, data_type=DataType.UINT32)
        spec.write_value(self._prob_command, data_type=DataType.S1615)
        spec.write_value(self._prob_in_change, data_type=DataType.S1615)

        # End-of-Spec:
        spec.end_specification()

    def get_recording_region_base_address(self, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._RECALL_REGIONS.RECORDING.value)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self) -> str:
        return "store_recall.aplx"
