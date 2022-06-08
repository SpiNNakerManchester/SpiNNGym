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

# common imports
from spinn_gym.games import SpinnGymApplicationVertex

# Bandit imports
from spinn_gym.games.multi_arm_bandit.bandit_machine_vertex import \
    BanditMachineVertex

NUMPY_DATA_ELEMENT_TYPE = numpy.double


# ----------------------------------------------------------------------------
# Bandit
# ----------------------------------------------------------------------------
class Bandit(SpinnGymApplicationVertex):

    BANDIT_REGION_BYTES = 4
    BASE_ARMS_REGION_BYTES = 11 * 4
    MAX_SIM_DURATION = 1000 * 60 * 60  # 1 hour

    # parameters expected by PyNN
    default_parameters = {
        'reward_delay': 200.0,
        'constraints': None,
        'rate_on': 20.0,
        'rate_off': 5.0,
        'constant_input': 0,
        'stochastic': 1,
        'reward_based': 1,
        'label': "Bandit",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION,
        'arms': [0.1, 0.9],
        'random_seed': [
            numpy.random.randint(10000), numpy.random.randint(10000),
            numpy.random.randint(10000), numpy.random.randint(10000)]}

    def __init__(self, arms=default_parameters['arms'],
                 reward_delay=default_parameters['reward_delay'],
                 reward_based=default_parameters['reward_based'],
                 rate_on=default_parameters['rate_on'],
                 rate_off=default_parameters['rate_off'],
                 stochastic=default_parameters['stochastic'],
                 constant_input=default_parameters['constant_input'],
                 constraints=default_parameters['constraints'],
                 label=default_parameters['label'],
                 simulation_duration_ms=default_parameters['duration'],
                 rand_seed=default_parameters['random_seed']):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label

        # Pass in variables
        self._arms = arms

        self._no_arms = len(arms)
        self._n_neurons = self._no_arms
        self._rand_seed = rand_seed

        self._reward_delay = reward_delay
        self._reward_based = reward_based

        self._rate_on = rate_on
        self._rate_off = rate_off
        self._stochastic = stochastic
        self._constant_input = constant_input

        # used to define size of recording region
        self._recording_size = int((simulation_duration_ms / 1000.) * 4)

        sdram_required = (
            self.BANDIT_REGION_BYTES + self.BASE_ARMS_REGION_BYTES +
            self._recording_size)

        # Superclasses
        super(Bandit, self).__init__(
            BanditMachineVertex(
                self._n_neurons, sdram_required, constraints, label, self,
                arms, reward_delay, reward_based, rate_on, rate_off,
                stochastic, constant_input, simulation_duration_ms, rand_seed),
            label=label, constraints=constraints)

        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        SimplePopulationSettable.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractAcceptsIncomingSynapses.__init__(self)
        self._change_requires_mapping = True

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self):
        return numpy.int32
