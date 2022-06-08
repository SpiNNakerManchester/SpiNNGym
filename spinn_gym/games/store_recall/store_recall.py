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
from pacman.model.graphs.common import Slice

# sPyNNaker imports
from spynnaker.pyNN.models.common import AbstractNeuronRecordable

# common imports
from spinn_gym.games import SpinnGymApplicationVertex

# Recall imports
from spinn_gym.games.store_recall.store_recall_machine_vertex import \
    RecallMachineVertex

NUMPY_DATA_ELEMENT_TYPE = numpy.double


class Bad_Table(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# ----------------------------------------------------------------------------
# Recall
# ----------------------------------------------------------------------------
class Recall(SpinnGymApplicationVertex):

    RECALL_REGION_BYTES = 4
    DATA_REGION_BYTES = 12 * 4
    MAX_SIM_DURATION = 1000 * 60 * 60 * 24  # 1 day

    # parameters expected by PyNN
    default_parameters = {
        'time_period': 200.0,
        'constraints': None,
        'rate_on': 50.0,
        'rate_off': 0.0,
        'pop_size': 1,
        'prob_command': 1.0/6.0,
        'prob_in_change': 1.0/2.0,
        'stochastic': 1,
        'reward': 0,
        'label': "Recall",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION,
        'random_seed': [
            numpy.random.randint(10000), numpy.random.randint(10000),
            numpy.random.randint(10000), numpy.random.randint(10000)]}

    def __init__(self,
                 rate_on=default_parameters['rate_on'],
                 rate_off=default_parameters['rate_off'],
                 pop_size=default_parameters['pop_size'],
                 prob_command=default_parameters['prob_command'],
                 prob_in_change=default_parameters['prob_in_change'],
                 time_period=default_parameters['time_period'],
                 stochastic=default_parameters['stochastic'],
                 reward=default_parameters['reward'],
                 constraints=default_parameters['constraints'],
                 label=default_parameters['label'],
                 simulation_duration_ms=default_parameters['duration'],
                 rand_seed=default_parameters['random_seed']):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label

        # Pass in variables
        self._rate_on = rate_on
        self._rate_off = rate_off
        self._stochastic = stochastic
        self._reward = reward
        self._pop_size = pop_size
        self._prob_command = prob_command
        self._prob_in_change = prob_in_change

        self._n_neurons = pop_size * 4
        self._rand_seed = rand_seed

        self._time_period = time_period

        # used to define size of recording region
        self._recording_size = int((simulation_duration_ms / 1000.) * 4)

        # technically as using OneAppOneMachine this is not necessary?
        sdram_required = (
            self.RECALL_REGION_BYTES + self.DATA_REGION_BYTES +
            self._recording_size)

        # Superclasses
        super(Recall, self).__init__(
            RecallMachineVertex(
                self._n_neurons, sdram_required, constraints, label, self,
                rate_on, rate_off, pop_size, prob_command, prob_in_change,
                time_period, stochastic, reward,
                simulation_duration_ms, rand_seed),
            label=label, constraints=constraints)

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
