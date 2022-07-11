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

from spinn_utilities.overrides import overrides

# common imports
from spinn_gym.games import SpinnGymApplicationVertex

# Logic imports
from spinn_gym.games.logic.logic_machine_vertex import LogicMachineVertex


class Bad_Table(Exception):
    """
    table and input sequence are not compatible
    """


# ----------------------------------------------------------------------------
# Logic
# ----------------------------------------------------------------------------
class Logic(SpinnGymApplicationVertex):

    ONE_DAY_IN_MS = 1000 * 60 * 60 * 24  # 1 day
    RANDOM_SEED = [numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000)]

    __slots__ = []

    def __init__(
            self, truth_table, input_sequence, rate_on=20.0, rate_off=5.0,
            score_delay=200.0, stochastic=1, constraints=None, label="Logic",
            simulation_duration_ms=ONE_DAY_IN_MS, random_seed=None):
        if random_seed is None:
            random_seed = list(self.RANDOM_SEED)

        n_neurons = len(input_sequence)
        if n_neurons != numpy.log2(len(truth_table)):
            try:
                raise Bad_Table('table and input sequence are not compatible')
            except Bad_Table as e:
                print("ERROR: ", e)
                # TODO is it safe to continue ??????
        machine_vertex = LogicMachineVertex(
            label, constraints, self, n_neurons, simulation_duration_ms,
            random_seed, truth_table, input_sequence, rate_on, rate_off,
            score_delay, stochastic)
        # Superclasses
        super(Logic, self).__init__(
            machine_vertex, label, constraints, n_neurons)

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self):
        return numpy.int32
