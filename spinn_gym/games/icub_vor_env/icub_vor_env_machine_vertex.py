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
import numpy

from spinn_utilities.overrides import overrides

from data_specification.enums.data_type import DataType

# PACMAN imports
from pacman.model.resources import ConstantSDRAM
from pacman.model.graphs.machine import MachineVertex

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
from spinn_front_end_common.utilities.exceptions import ConfigurationException

# sPyNNaker imports
from spynnaker.pyNN.data import SpynnakerDataView
from spynnaker.pyNN.utilities import constants

# spinn_gym imports
from spinn_gym.games.spinn_gym_machine_vertex import SpinnGymMachineVertex


# ----------------------------------------------------------------------------
# ICubVorEnvMachineVertex
# ----------------------------------------------------------------------------
class ICubVorEnvMachineVertex(SpinnGymMachineVertex):
    ICUB_VOR_ENV_REGION_BYTES = 4
    BASE_DATA_REGION_BYTES = 9 * 4
    # Probably better ways of doing this too, but keeping it for now
    RECORDABLE_VARIABLES = [
        "l_count", "r_count", "error", "eye_pos", "eye_vel"]
    RECORDABLE_DTYPES = [
        DataType.UINT32, DataType.UINT32, DataType.S1615, DataType.S1615,
        DataType.S1615]

    _ICUB_VOR_ENV_REGIONS = Enum(
        value="_ICUB_VOR_ENV_REGIONS",
        names=[('SYSTEM', 0),
               ('ICUB_VOR_ENV', 1),
               ('RECORDING', 2),
               ('DATA', 3)])

    def __init__(self, label, constraints, app_vertex, n_neurons,
                 simulation_duration_ms, random_seed,
                 head_pos, head_vel, perfect_eye_pos,
                 perfect_eye_vel, error_window_size, output_size, gain,
                 pos_to_vel, wta_decision, low_error_rate, high_error_rate):

        super(ICubVorEnvMachineVertex, self).__init__(
            label, constraints, app_vertex, n_neurons,
            self.ICUB_VOR_ENV_REGION_BYTES + self.BASE_DATA_REGION_BYTES,
            simulation_duration_ms, random_seed)

        # pass in variables
        self._head_pos = head_pos
        self._head_vel = head_vel
        self._perfect_eye_pos = perfect_eye_pos
        self._perfect_eye_vel = perfect_eye_vel
        self._error_window_size = error_window_size
        self._output_size = output_size
        self._gain = gain
        self._pos_to_vel = pos_to_vel
        self._wta_decision = wta_decision
        self._low_error_rate = low_error_rate
        self._high_error_rate = high_error_rate
        self._number_of_inputs = len(perfect_eye_pos)
        if self._number_of_inputs != len(perfect_eye_vel):
            raise ConfigurationException(
                "The length of perfect_eye_pos {} is not the same as the "
                "length of perfect_eye_vel {}".format(
                    self._number_of_inputs, len(perfect_eye_vel)))
        self._n_recordable_variables = len(self.RECORDABLE_VARIABLES)

        self._recording_size = int((simulation_duration_ms / error_window_size)
                                   * front_end_common_constants.BYTES_PER_WORD)

        self._sdram_required = ConstantSDRAM(
            self.ICUB_VOR_ENV_REGION_BYTES + self.BASE_DATA_REGION_BYTES +
            self._recording_size)

    @property
    @overrides(SpinnGymMachineVertex.sdram_required)
    def sdram_required(self):
        return self._sdram_required

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification)
    def generate_data_specification(self, spec, placement):
        vertex = placement.vertex

        spec.comment("\n*** Spec for ICubVorEnv Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=self._ICUB_VOR_ENV_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=self._ICUB_VOR_ENV_REGIONS.ICUB_VOR_ENV.value,
            size=self.ICUB_VOR_ENV_REGION_BYTES, label='ICubVorEnvParams')
        # reserve recording region
        spec.reserve_memory_region(
            self._ICUB_VOR_ENV_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(
                len(self.RECORDABLE_VARIABLES)))
        spec.reserve_memory_region(
            region=self._ICUB_VOR_ENV_REGIONS.DATA.value,
            size=self.BASE_DATA_REGION_BYTES + (self._number_of_inputs * 16),
            label='ICubVorEnvArms')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            self._ICUB_VOR_ENV_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name()))

        # Write icub_vor_env region containing routing key to transmit with
        spec.comment("\nWriting icub_vor_env region:\n")
        spec.switch_write_focus(
            self._ICUB_VOR_ENV_REGIONS.ICUB_VOR_ENV.value)
        routing_info = SpynnakerDataView.get_routing_infos()
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.LIVE_POISSON_CONTROL_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting icub_vor_env recording region:\n")
        spec.switch_write_focus(
            self._ICUB_VOR_ENV_REGIONS.RECORDING.value)
        recording_sizes = [
            self._recording_size for _ in range(self._n_recordable_variables)]
        spec.write_array(recording_utilities.get_recording_header_array(
            recording_sizes))

        # Write parameters for ICubVorEnv data
        spec.comment("\nWriting icub_vor_env data region:\n")
        float_scale = float(DataType.S1615.scale)
        spec.switch_write_focus(
            self._ICUB_VOR_ENV_REGIONS.DATA.value)
        spec.write_value(self._error_window_size, data_type=DataType.UINT32)
        spec.write_value(self._output_size, data_type=DataType.UINT32)
        spec.write_value(self._number_of_inputs, data_type=DataType.UINT32)
        spec.write_value(self.__round_to_nearest_accum(self._gain),
                         data_type=DataType.S1615)
        spec.write_value(self.__round_to_nearest_accum(self._pos_to_vel),
                         data_type=DataType.S1615)
        spec.write_value(int(self._wta_decision), data_type=DataType.UINT32)
        spec.write_value(self.__round_to_nearest_accum(self._low_error_rate),
                         data_type=DataType.S1615)
        spec.write_value(self.__round_to_nearest_accum(self._high_error_rate),
                         data_type=DataType.S1615)
        # Write the data - Arrays must be 32-bit values, so convert
        data = numpy.array(
            [int(x * float_scale) for x in self._perfect_eye_pos],
            dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))
        data = numpy.array(
            [int(x * float_scale) for x in self._perfect_eye_vel],
            dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))

        # End-of-Spec:
        spec.end_specification()

    def __round_to_nearest_accum(self, x):
        eps = 2. ** (-15)
        x_approx = numpy.floor((x / eps) + 0.5) * eps
        return x_approx

    @overrides(SpinnGymMachineVertex.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return [0, 1, 2, 3, 4]

    def get_recording_region_base_address(self, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._ICUB_VOR_ENV_REGIONS.RECORDING.value)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "icub_vor_env.aplx"

    @overrides(MachineVertex.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition_id):
        # If the vertex is controlling another vertex then the number of
        # keys needed is related to that vertex!
        if partition_id == constants.LIVE_POISSON_CONTROL_PARTITION_ID:
            partitions = SpynnakerDataView.\
                get_outgoing_edge_partitions_starting_at_vertex(
                    self.app_vertex)
            n_keys = 0
            for partition in partitions:
                if partition.identifier == partition_id:
                    for edge in partition.edges:
                        if edge.pre_vertex is not edge.post_vertex:
                            for m_vert in edge.post_vertex.machine_vertices:
                                n_keys += (
                                    m_vert.get_n_keys_for_partition(
                                        partition))
            return n_keys
        else:
            return (
                super(ICubVorEnvMachineVertex, self).get_n_keys_for_partition(
                    partition_id))
