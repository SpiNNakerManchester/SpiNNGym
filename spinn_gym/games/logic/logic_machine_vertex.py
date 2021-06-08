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
from pacman.executor.injection_decorator import inject_items
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ConstantSDRAM, ResourceContainer

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.interface.buffer_management.buffer_models import \
    AbstractReceiveBuffersToHost
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.utilities.utility_objs import ExecutableType
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


# ----------------------------------------------------------------------------
# LogicMachineVertex
# ----------------------------------------------------------------------------
class LogicMachineVertex(MachineVertex, AbstractGeneratesDataSpecification,
                         AbstractReceiveBuffersToHost,
                         AbstractHasAssociatedBinary):
    LOGIC_REGION_BYTES = 4
    BASE_DATA_REGION_BYTES = 9 * 4

    _LOGIC_REGIONS = Enum(
        value="_LOGIC_REGIONS",
        names=[('SYSTEM', 0),
               ('LOGIC', 1),
               ('RECORDING', 2),
               ('DATA', 3)])

    def __init__(self, vertex_slice, resources_required, constraints, label,
                 app_vertex, truth_table, input_sequence, rate_on, rate_off,
                 score_delay, stochastic, incoming_spike_buffer_size,
                 simulation_duration_ms, rand_seed):

        # resources required
        self._resources_required = ResourceContainer(
            sdram=ConstantSDRAM(resources_required))

        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label

        # Pass in variables
        self._truth_table = truth_table
        self._rate_on = rate_on
        self._rate_off = rate_off
        self._stochastic = stochastic
        self._input_sequence = input_sequence
        self._no_inputs = len(input_sequence)

        self._n_neurons = self._no_inputs
        self._rand_seed = rand_seed

        self._score_delay = score_delay

        # used to define size of recording region
        self._recording_size = int((simulation_duration_ms / 1000.) * 4)

        # Superclasses
        MachineVertex.__init__(
            self, label, constraints, app_vertex, vertex_slice)

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"routing_info": "MemoryRoutingInfos"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"routing_info"}
               )
    def generate_data_specification(self, spec, placement, routing_info):
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
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name()))

        # Write logic region containing routing key to transmit with
        spec.comment("\nWriting logic region:\n")
        spec.switch_write_focus(
            self._LOGIC_REGIONS.LOGIC.value)
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.SPIKE_PARTITION_ID))

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
        spec.write_value(self._rand_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[3], data_type=DataType.UINT32)
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

    @property
    def resources_required(self):
        return self._resources_required

    def get_minimum_buffer_sdram_usage(self):
        return 0  # probably should make this a user input

    def get_recorded_region_ids(self):
        return [0]

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._LOGIC_REGIONS.RECORDING.value, txrx)

    def get_n_keys_for_partition(self, partition):
        n_keys = 0
        # The way this has been written, there should only be one edge, but
        # better to be safe than sorry
        for edge in partition.edges:
            if edge.pre_vertex is not edge.post_vertex:
                n_keys += edge.post_vertex.get_n_keys_for_partition(partition)
        return n_keys

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "logic.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE
