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
from pacman.model.constraints.key_allocator_constraints \
    import ContiguousKeyRangeContraint
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)
from pacman.model.graphs.common import Slice
from spinn_utilities.config_holder import get_config_int

# SpiNNFrontEndCommon imports
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun
from spinn_front_end_common.abstract_models. \
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.data import FecDataView

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models \
    import AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

# Pendulum imports
from spinn_gym.games.inverted_pendulum.inverted_pendulum_machine_vertex \
    import PendulumMachineVertex

NUMPY_DATA_ELEMENT_TYPE = numpy.double


# ----------------------------------------------------------------------------
# Pendulum
# ----------------------------------------------------------------------------
class Pendulum(AbstractOneAppOneMachineVertex,
               AbstractProvidesOutgoingPartitionConstraints,
               AbstractAcceptsIncomingSynapses, AbstractNeuronRecordable,
               SimplePopulationSettable):

    @overrides(AbstractAcceptsIncomingSynapses.verify_splitter)
    def verify_splitter(self, splitter):
        # Need to ignore this verify
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(self, app_edge, synapse_info):

        # TODO: make this work properly (the following call does nothing)

        super(Pendulum, self).get_connections_from_machine(
            app_edge, synapse_info)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def clear_connection_cache(self):
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    PENDULUM_REGION_BYTES = 4
    BASE_DATA_REGION_BYTES = 15 * 4
    MAX_SIM_DURATION = 1000 * 60 * 60 * 24 * 7  # 1 week

    # parameters expected by PyNN
    default_parameters = {
        'constraints': None,
        'encoding': 0,  # 0 rate, 1 receptive bins, 2 spike time, 3 rank
        'time_increment': 20,
        'pole_length': 1.0,
        'pole_angle': 0.1,
        'reward_based': 1,
        'force_increments': 100,
        'max_firing_rate': 100,
        'number_of_bins': 20,
        'central': 1,
        'bin_overlap': 2,
        'tau_force': 0,
        'label': "pole",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION,
        'rand_seed': [0, 1, 2, 3],
        }

    def __init__(self, constraints=default_parameters['constraints'],
                 encoding=default_parameters['encoding'],
                 time_increment=default_parameters['time_increment'],
                 pole_length=default_parameters['pole_length'],
                 pole_angle=default_parameters['pole_angle'],
                 reward_based=default_parameters['reward_based'],
                 force_increments=default_parameters['force_increments'],
                 max_firing_rate=default_parameters['max_firing_rate'],
                 number_of_bins=default_parameters['number_of_bins'],
                 central=default_parameters['central'],
                 rand_seed=default_parameters['rand_seed'],
                 bin_overlap=default_parameters['bin_overlap'],
                 tau_force=default_parameters['tau_force'],
                 label=default_parameters['label'],
                 incoming_spike_buffer_size=default_parameters[
                     'incoming_spike_buffer_size'],
                 simulation_duration_ms=default_parameters['duration']):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label

        self._encoding = encoding

        # Pass in variables
        self._pole_length = pole_length
        self._pole_angle = pole_angle

        self._force_increments = force_increments
        # for rate based it's only 1 neuron per metric
        # (position, angle, velocity of both)
        if self._encoding == 0:
            self._n_neurons = 4
        else:
            self._n_neurons = 4 * number_of_bins

        self._time_increment = time_increment
        self._reward_based = reward_based

        self._max_firing_rate = max_firing_rate
        self._number_of_bins = number_of_bins
        self._central = central
        self._rand_seed = rand_seed
        self._bin_overlap = bin_overlap
        self._tau_force = tau_force

        # used to define size of recording region
        self._recording_size = int((simulation_duration_ms / 1000.) * 4)

        # technically as using OneAppOneMachine this is not necessary?
        resources_required = (
            self.PENDULUM_REGION_BYTES + self.BASE_DATA_REGION_BYTES +
            self._recording_size)

        vertex_slice = Slice(0, self._n_neurons - 1)

        # Superclasses
        super(Pendulum, self).__init__(
            PendulumMachineVertex(
                vertex_slice, resources_required, constraints, label, self,
                encoding, time_increment, pole_length, pole_angle,
                reward_based, force_increments, max_firing_rate,
                number_of_bins, central, bin_overlap, tau_force,
                incoming_spike_buffer_size, simulation_duration_ms, rand_seed
                ),
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
    @overrides(AbstractNeuronRecordable.clear_recording)
    def clear_recording(self, variable):
        self._clear_recording_region(0)

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
        return 10000  # 10 seconds hard coded in inverted_pendulum.c

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(self, variable):
        vertex = self.machine_vertices.pop()
        placement = FecDataView.get_placement_of_vertex(vertex)

        # Read the data recorded
        buffer_manager = FecDataView.get_buffer_manager()
        data_values, _ = buffer_manager.get_data_by_placement(placement, 0)
        data = data_values

        numpy_format = list()
        numpy_format.append(("Score", numpy.float32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

        # return formatted_data
        return output_data

    def _clear_recording_region(self, recording_region_id):
        """ Clear a recorded data region from the buffer manager.

        :param buffer_manager: the buffer manager object
        :param placements: the placements object
        :param recording_region_id: the recorded region ID for clearing
        :rtype: None
        """
        buffer_manager = FecDataView.get_buffer_manager()
        for machine_vertex in self.machine_vertices:
            placement = FecDataView.get_placement_of_vertex(machine_vertex)
            buffer_manager.clear_recorded_data(
                placement.x, placement.y, placement.p, recording_region_id)

    def reset_ring_buffer_shifts(self):
        pass

    def __str__(self):
        return "{} with {} atoms".format(self._label, self.n_atoms)

    def __repr__(self):
        return self.__str__()
