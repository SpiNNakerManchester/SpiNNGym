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

# Pendulum imports
from spinn_gym.games.inverted_pendulum.inverted_pendulum_machine_vertex \
    import PendulumMachineVertex


# ----------------------------------------------------------------------------
# Pendulum
# ----------------------------------------------------------------------------
class Pendulum(SpinnGymApplicationVertex):

    ONE_WEEK_IN_MS = 1000 * 60 * 60 * 24 * 7  # 1 week
    RANDOM_SEED = [0, 1, 2, 3]

    __slots__ = []

    def __init__(self, constraints=None, encoding=0, time_increment=20,
                 pole_length=1.0, pole_angle=0.1, reward_based=1,
                 force_increments=100, max_firing_rate=100,
                 number_of_bins=20, central=1, random_seed=None,
                 bin_overlap=2, tau_force=0, label="pole",
                 simulation_duration_ms=ONE_WEEK_IN_MS):
        """

        :param constraints:
        :param encoding: 0 rate, 1 receptive bins, 2 spike time, 3 rank
        :param time_increment:
        :param pole_length:
        :param pole_angle:
        :param reward_based:
        :param force_increments:
        :param max_firing_rate:
        :param number_of_bins:
        :param central:
        :param random_seed:
        :param bin_overlap:
        :param tau_force:
        :param label:
        :param simulation_duration_ms:
        """
        if random_seed is None:
            random_seed = list(self.RANDOM_SEED)

        # for rate based it's only 1 neuron per metric
        # (position, angle, velocity of both)
        if encoding == 0:
            n_neurons = 4
        else:
            n_neurons = 4 * number_of_bins

        machine_vertex =  PendulumMachineVertex(
            label, constraints, self, n_neurons,
            simulation_duration_ms, random_seed,
            encoding, time_increment, pole_length, pole_angle,
            reward_based, force_increments, max_firing_rate,
            number_of_bins, central, bin_overlap, tau_force)

        # Superclasses
        super(Pendulum, self).__init__(
           machine_vertex, label, constraints, n_neurons)

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self):
        return numpy.float32
