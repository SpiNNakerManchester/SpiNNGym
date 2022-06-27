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

from spinn_utilities.abstract_base import abstractproperty
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

    __slots__ = [
        # list of 4 numbers to be the random seeds for the c code
        "_random_seed",
        # size of recording region
        "_recording_size",
        # resources needed for this vertex
        "_resources_required"]

    def __init__(self, label, constraints, app_vertex, n_neurons,
                 region_bytes, simulation_duration_ms, random_seed):
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

        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        :raises PacmanValueError: If the slice of the machine_vertex is too big
        :raise AttributeError:
            If a not None app_vertex is not an ApplicationVertex

        """

        vertex_slice = Slice(0, n_neurons - 1)

        # Superclasses
        MachineVertex.__init__(
            self, label, constraints, app_vertex, vertex_slice)

        # Define size of recording region
        self._recording_size = int((simulation_duration_ms/10000.) * 4)
        # TODO Should this not round UP!

        self._resources_required = ResourceContainer(
            sdram=ConstantSDRAM(region_bytes + self._recording_size))

        self._random_seed = random_seed

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return self._resources_required

    @overrides(MachineVertex.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, _partition):
        n_keys = 0
        # The way most of these vertices are written, there should only be one
        # edge, but better to be safe than sorry
        for edge in _partition.edges:
            if edge.pre_vertex is not edge.post_vertex:
                n_keys += edge.post_vertex.get_n_keys_for_partition(_partition)
        return n_keys

    @abstractproperty
    def get_recording_region_base_address(self, placement):
        """
        The recording region base address
        """

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return [0]

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE
