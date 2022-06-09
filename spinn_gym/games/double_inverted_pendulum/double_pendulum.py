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
from spinn_gym.games.double_inverted_pendulum.double_pendulum_machine_vertex \
    import DoublePendulumMachineVertex


# ----------------------------------------------------------------------------
# Double Pendulum
# ----------------------------------------------------------------------------
class DoublePendulum(SpinnGymApplicationVertex):
    ONE_WEEK_IN_MS = 1000 * 60 * 60 * 24 * 7  # 1 week
    RANDOM_SEED = [0, 1, 2, 3]

    __slots__ = []

    def __init__(
            self, constraints=None, encoding=0, time_increment=20,
            pole_length=1.0, pole_angle=0.1, pole2_length=0, pole2_angle=0,
            reward_based=1, force_increments=100, max_firing_rate=100,
            number_of_bins=20, central=1, random_seed=None, bin_overlap=2,
            tau_force=0, label="pole", simulation_duration_ms=ONE_WEEK_IN_MS):
        """

        :param constraints:
        :param encoding:  0 rate, 1 receptive bins, 2 spike time, 3 rank
        :param time_increment:
        :param pole_length:
        :param pole_angle:
        :param pole2_length:
        :param pole2_angle:
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
        n_neurons = 6 * number_of_bins

        # Superclasses
        super(DoublePendulum, self).__init__(
            DoublePendulumMachineVertex(
                n_neurons, constraints, label, self,
                encoding, time_increment, pole_length, pole_angle,
                pole2_length, pole2_angle, reward_based, force_increments,
                max_firing_rate, number_of_bins, central, bin_overlap,
                tau_force, simulation_duration_ms, random_seed),
            label=label, constraints=constraints, n_atoms=n_neurons)

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self):
        return numpy.float32
