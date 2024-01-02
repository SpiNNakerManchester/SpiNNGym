# Copyright (c) 2019 The University of Manchester
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy

from spinn_utilities.overrides import overrides

# common imports
from spinn_gym.games import SpinnGymApplicationVertex

# Recall imports
from spinn_gym.games.store_recall.store_recall_machine_vertex import \
    RecallMachineVertex


# ----------------------------------------------------------------------------
# Recall
# ----------------------------------------------------------------------------
class Recall(SpinnGymApplicationVertex):

    ONE_DAY_IN_MS = 1000 * 60 * 60 * 24  # 1 day
    RANDOM_SEED = [numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000)]

    __slots__ = []

    def __init__(
            self, rate_on=50.0, rate_off=0.0, pop_size=1, prob_command=1.0/6.0,
            prob_in_change=1.0/2.0, time_period=200.0, stochastic=1,
            reward=0, label="Recall",
            simulation_duration_ms=ONE_DAY_IN_MS,  random_seed=None):
        if random_seed is None:
            random_seed = list(self.RANDOM_SEED)

        n_neurons = pop_size * 4

        machine_vertex = RecallMachineVertex(
            label, self, n_neurons, simulation_duration_ms, random_seed,
            rate_on, rate_off, pop_size, prob_command,
            prob_in_change, time_period, stochastic, reward)
        # Superclasses
        super(Recall, self).__init__(machine_vertex, label, n_neurons)

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self) -> type:
        return numpy.int32
