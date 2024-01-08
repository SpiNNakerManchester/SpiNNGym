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

# Bandit imports
from spinn_gym.games.multi_arm_bandit.bandit_machine_vertex import \
    BanditMachineVertex


# ----------------------------------------------------------------------------
# Bandit
# ----------------------------------------------------------------------------
class Bandit(SpinnGymApplicationVertex):
    ONE_DAY_IN_MS = 1000 * 60 * 60 * 24  # 1 day
    RANDOM_SEED = [numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000)]
    ARMS = [0.1, 0.9]

    __slots__ = ()

    def __init__(self, arms=None, reward_delay=200.0, reward_based=1,
                 rate_on=20.0, rate_off=5.0, stochastic=1,
                 constant_input=0, label="Bandit",
                 simulation_duration_ms=ONE_DAY_IN_MS, random_seed=None):
        if arms is None:
            arms = list(self.ARMS)
        if random_seed is None:
            random_seed = list(self.RANDOM_SEED)

        n_neurons = len(arms)

        machine_vertex = BanditMachineVertex(
            label, self, n_neurons, simulation_duration_ms, random_seed,
            arms, reward_delay, reward_based, rate_on,
            rate_off, stochastic, constant_input)

        # Superclasses
        super(Bandit, self).__init__(machine_vertex, label, n_neurons)

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self) -> type:
        return numpy.int32
