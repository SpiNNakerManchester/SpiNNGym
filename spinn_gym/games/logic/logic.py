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

import numpy

from spinn_utilities.overrides import overrides

# PACMAN imports
from pacman.model.constraints.key_allocator_constraints import \
    ContiguousKeyRangeContraint
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)
from pacman.model.graphs.common import Slice
from spinn_utilities.config_holder import get_config_int

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

# Logic imports
from spinn_gym.games.logic.logic_machine_vertex import LogicMachineVertex

NUMPY_DATA_ELEMENT_TYPE = numpy.double


class Bad_Table(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# ----------------------------------------------------------------------------
# Logic
# ----------------------------------------------------------------------------
class Logic(AbstractOneAppOneMachineVertex,
            AbstractProvidesOutgoingPartitionConstraints,
            AbstractAcceptsIncomingSynapses, AbstractNeuronRecordable,
            SimplePopulationSettable):

    @overrides(AbstractAcceptsIncomingSynapses.verify_splitter)
    def verify_splitter(self, splitter):
        # Need to ignore this verify
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(
            self, transceiver, placements, app_edge, synapse_info):

        # TODO: make this work properly (the following call does nothing)

        super(Logic, self).get_connections_from_machine(
            transceiver, placements, app_edge, synapse_info)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def clear_connection_cache(self):
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    LOGIC_REGION_BYTES = 4
    BASE_DATA_REGION_BYTES = 9 * 4
    MAX_SIM_DURATION = 1000 * 60 * 60 * 24  # 1 day

    # parameters expected by PyNN
    default_parameters = {
        'score_delay': 200.0,
        'constraints': None,
        'rate_on': 20.0,
        'rate_off': 5.0,
        'input_sequence': [0, 1],
        'stochastic': 1,
        'label': "Logic",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION,
        'truth_table': [0, 1, 1, 0],
        'random_seed': [
            numpy.random.randint(10000), numpy.random.randint(10000),
            numpy.random.randint(10000), numpy.random.randint(10000)]}

    def __init__(self, truth_table, input_sequence,
                 rate_on=default_parameters['rate_on'],
                 rate_off=default_parameters['rate_off'],
                 score_delay=default_parameters['score_delay'],
                 stochastic=default_parameters['stochastic'],
                 constraints=default_parameters['constraints'],
                 label=default_parameters['label'],
                 incoming_spike_buffer_size=default_parameters[
                     'incoming_spike_buffer_size'],
                 simulation_duration_ms=default_parameters['duration'],
                 rand_seed=default_parameters['random_seed']):
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
        if self._no_inputs != numpy.log2(len(self._truth_table)):
            try:
                raise Bad_Table('table and input sequence are not compatible')
            except Bad_Table as e:
                print("ERROR: ", e)

        self._n_neurons = self._no_inputs
        self._rand_seed = rand_seed

        self._score_delay = score_delay

        # used to define size of recording region
        self._recording_size = int((simulation_duration_ms / 1000.) * 4)

        # (static) resources required
        # technically as using OneAppOneMachine this is not necessary?
        resources_required = (
            self.LOGIC_REGION_BYTES + self.BASE_DATA_REGION_BYTES +
            self._recording_size)

        vertex_slice = Slice(0, self._n_neurons - 1)

        # Superclasses
        super(Logic, self).__init__(
            LogicMachineVertex(
                vertex_slice, resources_required, constraints, label, self,
                truth_table, input_sequence, rate_on, rate_off, score_delay,
                stochastic, incoming_spike_buffer_size, simulation_duration_ms,
                rand_seed),
            label=label, constraints=constraints)

        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        SimplePopulationSettable.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractAcceptsIncomingSynapses.__init__(self)
        self._change_requires_mapping = True
        if incoming_spike_buffer_size is None:
            self._incoming_spike_buffer_size = get_config_int(
                "Simulation", "incoming_spike_buffer_size")

    def neurons(self):
        return self._n_neurons

    @property
    @overrides(AbstractOneAppOneMachineVertex.n_atoms)
    def n_atoms(self):
        return self._n_neurons

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
        self._change_requires_neuron_parameters_reload = True

    # ------------------------------------------------------------------------
    # Recording overrides
    # ------------------------------------------------------------------------
    @overrides(
        AbstractNeuronRecordable.clear_recording)
    def clear_recording(
            self, variable, buffer_manager, placements):
        self._clear_recording_region(buffer_manager, placements, 0)

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

    @overrides(AbstractNeuronRecordable.get_neuron_sampling_interval)
    def get_neuron_sampling_interval(self, variable):
        return 10000  # 10 seconds hard coded in logic.c

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(
            self, variable, n_machine_time_steps, placements, buffer_manager):
        vertex = self.machine_vertices.pop()
        placement = placements.get_placement_of_vertex(vertex)

        # Read the data recorded
        data_values, _ = buffer_manager.get_data_by_placement(placement, 0)
        data = data_values

        numpy_format = list()
        numpy_format.append(("Score", numpy.int32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

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

    def __str__(self):
        return "{} with {} atoms".format(self._label, self.n_atoms)

    def __repr__(self):
        return self.__str__()
