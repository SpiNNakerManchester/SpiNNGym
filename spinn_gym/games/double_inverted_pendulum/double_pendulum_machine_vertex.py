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
# DoublePendulumMachineVertex
# ----------------------------------------------------------------------------
class DoublePendulumMachineVertex(SpinnGymMachineVertex):
    PENDULUM_REGION_BYTES = 4
    DATA_REGION_BYTES = 17 * 4

    __slots__ = (
        "_bin_overlap", "_central", "_encoding", "_force_increments",
        "_max_firing_rate", "_number_of_bins", "_pole_angle", "_pole2_angle",
        "_pole_length", "_pole2_length", "_reward_based", "_tau_force",
        "_time_increment")

    _DOUBLE_PENDULUM_REGIONS = Enum(
        value="_DOUBLE_PENDULUM_REGIONS",
        names=[('SYSTEM', 0),
               ('PENDULUM', 1),
               ('RECORDING', 2),
               ('DATA', 3)])

    def __init__(
            self, label, app_vertex, n_neurons,
            simulation_duration_ms, random_seed,
            encoding, time_increment, pole_length, pole_angle, pole2_length,
            pole2_angle, reward_based, force_increments, max_firing_rate,
            number_of_bins, central, bin_overlap, tau_force):
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
        :param encoding:
        :param time_increment:
        :param pole_length:
        :param pole_angle:
        :param pole2_length:
        :param pole2_angle:
        :param reward_based:
        :param force_increments:
        :param max_firing_rate:
        :param number_of_bins:
        :param central:
        :param bin_overlap:
        :param tau_force:

        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        :raises PacmanValueError: If the slice of the machine_vertex is too big
        :raise AttributeError:
            If a not None app_vertex is not an ApplicationVertex
        """

        # Superclasses
        super(DoublePendulumMachineVertex, self).__init__(
            label, app_vertex, n_neurons,
            self.PENDULUM_REGION_BYTES + self.DATA_REGION_BYTES,
            simulation_duration_ms,  random_seed)

        self._encoding = encoding

        # Pass in variables
        self._pole_length = pole_length
        self._pole_angle = pole_angle
        self._pole2_length = pole2_length
        self._pole2_angle = pole2_angle

        self._force_increments = force_increments
        self._time_increment = time_increment
        self._reward_based = reward_based

        self._max_firing_rate = max_firing_rate
        self._number_of_bins = number_of_bins
        self._central = central
        self._bin_overlap = bin_overlap
        self._tau_force = tau_force

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @overrides(SpinnGymMachineVertex.generate_data_specification)
    def generate_data_specification(
            self, spec: DataSpecificationGenerator, placement: Placement):
        # pylint: disable=arguments-differ
        vertex = placement.vertex

        spec.comment("\n*** Spec for Double Pendulum Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=self._DOUBLE_PENDULUM_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=self._DOUBLE_PENDULUM_REGIONS.PENDULUM.value,
            size=self.PENDULUM_REGION_BYTES, label='PendulumVertex')
        # reserve recording region
        spec.reserve_memory_region(
            self._DOUBLE_PENDULUM_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=self._DOUBLE_PENDULUM_REGIONS.DATA.value,
            size=self.DATA_REGION_BYTES, label='PendulumData')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            self._DOUBLE_PENDULUM_REGIONS.SYSTEM.value)
        assert isinstance(vertex, AbstractHasAssociatedBinary)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name()))

        # Write pendulum region containing routing key to transmit with
        spec.comment("\nWriting double pendulum region:\n")
        spec.switch_write_focus(
            self._DOUBLE_PENDULUM_REGIONS.PENDULUM.value)
        routing_info = SpynnakerDataView.get_routing_infos()
        spec.write_value(routing_info.get_single_first_key_from_pre_vertex(
            vertex))

        # Write recording region for score
        spec.comment("\nWriting double pendulum recording region:\n")
        spec.switch_write_focus(
            self._DOUBLE_PENDULUM_REGIONS.RECORDING.value)
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size]))

        # Write probabilites for arms
        spec.comment("\nWriting double pendulum data region:\n")
        spec.switch_write_focus(
            self._DOUBLE_PENDULUM_REGIONS.DATA.value)
        spec.write_value(self._encoding, data_type=DataType.UINT32)
        spec.write_value(self._time_increment, data_type=DataType.UINT32)
        spec.write_value(self._pole_length, data_type=DataType.S1615)
        spec.write_value(self._pole_angle, data_type=DataType.S1615)
        spec.write_value(self._pole2_length, data_type=DataType.S1615)
        spec.write_value(self._pole2_angle, data_type=DataType.S1615)
        spec.write_value(self._reward_based, data_type=DataType.UINT32)
        spec.write_value(self._force_increments, data_type=DataType.UINT32)
        spec.write_value(self._max_firing_rate, data_type=DataType.UINT32)
        spec.write_value(self._number_of_bins, data_type=DataType.UINT32)
        spec.write_value(self._central, data_type=DataType.UINT32)
        spec.write_value(self._random_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[3], data_type=DataType.UINT32)
        spec.write_value(self._bin_overlap, data_type=DataType.S1615)
        spec.write_value(self._tau_force, data_type=DataType.S1615)

        # End-of-Spec:
        spec.end_specification()

    @overrides(SpinnGymMachineVertex.get_recording_region_base_address)
    def get_recording_region_base_address(self, placement: Placement) -> int:
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._DOUBLE_PENDULUM_REGIONS.RECORDING.value)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self) -> str:
        return "double_inverted_pendulum.aplx"
