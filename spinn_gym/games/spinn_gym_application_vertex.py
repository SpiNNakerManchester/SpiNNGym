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

import numpy

from spinn_utilities.abstract_base import abstractproperty
from spinn_utilities.overrides import overrides

# PACMAN imports
from pacman.model.constraints.key_allocator_constraints import \
    ContiguousKeyRangeContraint
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)

# SpinnFrontEndCommon imports
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun
from spinn_front_end_common.abstract_models. \
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import \
    AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable


class SpinnGymApplicationVertex(
        AbstractOneAppOneMachineVertex,
        AbstractProvidesOutgoingPartitionConstraints,
        AbstractAcceptsIncomingSynapses, AbstractNeuronRecordable,
        SimplePopulationSettable):

    __slots__ = [
        # A flag to detect a reset must be hard
        "_change_requires_mapping"]

    def __init__(self, machine_vertex, label, constraints, n_atoms):
        """
        Creates an ApplicationVertex which has exactly one predefined \
        MachineVertex

        :param machine_vertex: MachineVertex
        :param str label: The optional name of the vertex.
        :param constraints:
            The optional initial constraints of the vertex.
        :type constraints: iterable(AbstractConstraint) or None
        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        """
        super(SpinnGymApplicationVertex, self).__init__(
            machine_vertex, label, constraints, n_atoms)

        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        SimplePopulationSettable.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractAcceptsIncomingSynapses.__init__(self)
        self._change_requires_mapping = True

    @overrides(AbstractAcceptsIncomingSynapses.verify_splitter)
    def verify_splitter(self, splitter):
        # See https://github.com/SpiNNakerManchester/sPyNNaker/issues/1192
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(
            self, transceiver, placements, app_edge, synapse_info):

        # TODO: make this work properly (the following call does nothing)

        super(SpinnGymApplicationVertex, self).get_connections_from_machine(
            transceiver, placements, app_edge, synapse_info)

    @overrides(AbstractAcceptsIncomingSynapses.set_synapse_dynamics)
    def set_synapse_dynamics(self, synapse_dynamics):
        pass
        # TODO Should this be a pass or a NotImplemented ?

    @overrides(AbstractAcceptsIncomingSynapses.clear_connection_cache)
    def clear_connection_cache(self):
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    # ------------------------------------------------------------------------
    # AbstractProvidesOutgoingPartitionConstraints overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractProvidesOutgoingPartitionConstraints.
               get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return [ContiguousKeyRangeContraint()]

    @property
    @overrides(AbstractChangableAfterRun.requires_mapping)
    def requires_mapping(self):
        return self._change_requires_mapping

    @overrides(AbstractChangableAfterRun.mark_no_changes)
    def mark_no_changes(self):
        self._change_requires_mapping = False

    @overrides(SimplePopulationSettable.set_value)
    def set_value(self, key, value):
        SimplePopulationSettable.set_value(self, key, value)
        # TODO there was a AbstractRewritesDataSpecification variable here?

    # ------------------------------------------------------------------------
    # Recording overrides
    # ------------------------------------------------------------------------
    @overrides(
        AbstractNeuronRecordable.clear_recording)
    def clear_recording(self, variable, buffer_manager, placements):
        for machine_vertex in self.machine_vertices:
            placement = placements.get_placement_of_vertex(machine_vertex)
            buffer_manager.clear_recorded_data(
                placement.x, placement.y, placement.p, 0)

    @overrides(AbstractNeuronRecordable.get_recordable_variables)
    def get_recordable_variables(self):
        return 'score'

    @overrides(AbstractNeuronRecordable.is_recording)
    def is_recording(self, variable):
        return True

    @overrides(AbstractNeuronRecordable.set_recording)
    def set_recording(self, variable, new_state=True, sampling_interval=None,
                      indexes=None):
        pass
        # TODO Should this be a implemented, pass or a NotImplemented ?

    @overrides(AbstractNeuronRecordable.get_neuron_sampling_interval)
    def get_neuron_sampling_interval(self, variable):
        return 10000  # 10 seconds hard coded in bkout.c

    @abstractproperty
    def score_format(self):
        """
        The numpy format for the scores data
        """

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(
            self, variable, n_machine_time_steps, placements, buffer_manager):
        vertex = self.machine_vertices.pop()
        placement = placements.get_placement_of_vertex(vertex)

        # Read the data recorded
        data_values, _ = buffer_manager.get_data_by_placement(placement, 0)
        data = data_values

        numpy_format = list()
        numpy_format.append(("Score", self.score_format))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

        # return formatted_data
        return output_data

    def _clear_recording_region(
            self, buffer_manager, placements, recording_region_id):
        """ Clear a recorded data region from the buffer manager.

        :param buffer_manager: the buffer manager object
        :param placements: the placements object
        :param recording_region_id: the recorded region ID for clearing
        :rtype: None
        """
        for machine_vertex in self.machine_vertices:
            placement = placements.get_placement_of_vertex(machine_vertex)
            buffer_manager.clear_recorded_data(
                placement.x, placement.y, placement.p, recording_region_id)

    def reset_ring_buffer_shifts(self):
        pass
        # TODO Should this be a pass or a NotImplemented ?

    def __str__(self):
        return "{} with {} atoms".format(self._label, self.n_atoms)

    def __repr__(self):
        return self.__str__()
