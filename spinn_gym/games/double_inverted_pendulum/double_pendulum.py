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

NUMPY_DATA_ELEMENT_TYPE = numpy.double


# ----------------------------------------------------------------------------
# Double Pendulum
# ----------------------------------------------------------------------------
class DoublePendulum(SpinnGymApplicationVertex):

    PENDULUM_REGION_BYTES = 4
    BASE_DATA_REGION_BYTES = 17 * 4
    MAX_SIM_DURATION = 1000 * 60 * 60 * 24 * 7  # 1 week

    # parameters expected by PyNN
    default_parameters = {
        'constraints': None,
        'encoding': 0,  # 0 rate, 1 receptive bins, 2 spike time, 3 rank
        'time_increment': 20,
        'pole_length': 1.0,
        'pole_angle': 0.1,
        'pole2_length': 0.1,
        'pole2_angle': 0,
        'reward_based': 1,
        'force_increments': 100,
        'max_firing_rate': 100,
        'number_of_bins': 20,
        'central': 1,
        'rand_seed': [0, 1, 2, 3],
        'bin_overlap': 2,
        'tau_force': 0,
        'label': "pole",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION}

    # **HACK** for Projection to connect a synapse type is required

    def __init__(self, constraints=default_parameters['constraints'],
                 encoding=default_parameters['encoding'],
                 time_increment=default_parameters['time_increment'],
                 pole_length=default_parameters['pole_length'],
                 pole_angle=default_parameters['pole_angle'],
                 pole2_length=default_parameters['pole2_length'],
                 pole2_angle=default_parameters['pole2_angle'],
                 reward_based=default_parameters['reward_based'],
                 force_increments=default_parameters['force_increments'],
                 max_firing_rate=default_parameters['max_firing_rate'],
                 number_of_bins=default_parameters['number_of_bins'],
                 central=default_parameters['central'],
                 rand_seed=default_parameters['rand_seed'],
                 bin_overlap=default_parameters['bin_overlap'],
                 tau_force=default_parameters['tau_force'],
                 label=default_parameters['label'],
                 simulation_duration_ms=default_parameters['duration']):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label

        self._encoding = encoding

        # Pass in variables
        self._pole_length = pole_length
        self._pole_angle = pole_angle
        self._pole2_length = pole2_length
        self._pole2_angle = pole2_angle

        self._force_increments = force_increments
        # for rate based it's only 1 neuron per metric
        # (position, angle, velocity of both)
        self._n_neurons = 6 * number_of_bins

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
        sdram_required = (
            self.PENDULUM_REGION_BYTES + self.BASE_DATA_REGION_BYTES +
            self._recording_size)

        # Superclasses
        super(DoublePendulum, self).__init__(
            DoublePendulumMachineVertex(
                self._n_neurons, sdram_required, constraints, label, self,
                encoding, time_increment, pole_length, pole_angle,
                pole2_length, pole2_angle, reward_based, force_increments,
                max_firing_rate, number_of_bins, central, bin_overlap,
                tau_force, simulation_duration_ms, rand_seed),
            label=label, constraints=constraints)

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self):
        return numpy.float32
