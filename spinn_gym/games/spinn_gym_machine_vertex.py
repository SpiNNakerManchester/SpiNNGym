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

from spinn_utilities.overrides import overrides


# PACMAN imports
from pacman.model.graphs.common import Slice

from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ConstantSDRAM, ResourceContainer

# SpinnFrontEndCommon imports
from spinn_front_end_common.interface.buffer_management.buffer_models.\
    abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.abstract_models.\
    abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary


class SpinnGymMachineVertex(MachineVertex, AbstractGeneratesDataSpecification,
                            AbstractReceiveBuffersToHost,
                            AbstractHasAssociatedBinary):

    def __init__(self, label, constraints, app_vertex, n_neurons,
                 sdram_required, simulation_duration_ms, rand_seed):

        vertex_slice = Slice(0, n_neurons - 1)

        # Superclasses
        MachineVertex.__init__(
            self, label, constraints, app_vertex, vertex_slice)

        self._resources_required = ResourceContainer(
            sdram=ConstantSDRAM(sdram_required))

        self._rand_seed = rand_seed

        # Define size of recording region
        self._recording_size = int((simulation_duration_ms/10000.) * 4)


    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return self._resources_required

    @overrides(MachineVertex.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, _partition):
        n_keys = 0
        # The way this has been written, there should only be one edge, but
        # better to be safe than sorry
        for edge in _partition.edges:
            if edge.pre_vertex is not edge.post_vertex:
                n_keys += edge.post_vertex.get_n_keys_for_partition(_partition)
        return n_keys

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return [0]

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        # return ExecutableStartType.USES_SIMULATION_INTERFACE
        return ExecutableType.USES_SIMULATION_INTERFACE
