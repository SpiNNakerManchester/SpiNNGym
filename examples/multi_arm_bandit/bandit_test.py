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

import spynnaker8 as p
import spinn_gym as gym

from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from spinn_utilities.config_holder import get_config_int
from spinn_front_end_common.utilities.globals_variables import get_simulator


def get_scores(bandit_pop, simulator):
    b_vertex = bandit_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager,
        get_config_int("Machine", "machine_time_step"))
    return scores.tolist()


p.setup(timestep=1.0)

probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

input_size = len(probabilities)
input_pop = p.Population(input_size, p.SpikeSourcePoisson(rate=5))

output_pop1 = p.Population(2, p.IF_cond_exp())
output_pop2 = p.Population(2, p.IF_cond_exp())

random_seed = []
for j in range(4):
    random_seed.append(np.random.randint(0xffff))
arms_pop = p.Population(input_size, gym.Bandit(
    arms=probabilities, reward_delay=500, rand_seed=random_seed, stochastic=0))

input_pop.record('spikes')
# arms_pop.record('spikes')
output_pop1.record('spikes')
output_pop2.record('spikes')

i2a = p.Projection(input_pop, arms_pop, p.AllToAllConnector())

# neuron ID 0 = reward
# neuron ID 1 = no reward
test_rec = p.Projection(arms_pop, arms_pop, p.AllToAllConnector(),
                        p.StaticSynapse(weight=0.1, delay=0.5))
i2o1 = p.Projection(arms_pop, output_pop1, p.AllToAllConnector(),
                    p.StaticSynapse(weight=0.1, delay=0.5))
i2o2 = p.Projection(arms_pop, output_pop2, p.OneToOneConnector(),
                    p.StaticSynapse(weight=0.1, delay=0.5))

simulator = get_simulator()

runtime = 10000
p.run(runtime)

scores = get_scores(bandit_pop=arms_pop, simulator=simulator)

print(scores)

spikes_in = input_pop.get_data('spikes').segments[0].spiketrains
spikes_out1 = output_pop1.get_data('spikes').segments[0].spiketrains
spikes_out2 = output_pop2.get_data('spikes').segments[0].spiketrains
Figure(
    Panel(spikes_in, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_out1, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_out2, xlabel="Time (ms)", ylabel="nID", xticks=True)
)
plt.show()

p.end()
