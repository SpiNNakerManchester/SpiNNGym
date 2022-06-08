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

# common imports
from spinn_gym.games import SpinnGymApplicationVertex

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
class Logic(SpinnGymApplicationVertex):

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
        sdram_required = (
            self.LOGIC_REGION_BYTES + self.BASE_DATA_REGION_BYTES +
            self._recording_size)

        # Superclasses
        super(Logic, self).__init__(
            LogicMachineVertex(
                self._n_neurons, sdram_required, constraints, label, self,
                truth_table, input_sequence, rate_on, rate_off, score_delay,
                stochastic, simulation_duration_ms, rand_seed),
            label=label, constraints=constraints)

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self):
        return numpy.int32
