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

import matplotlib.pyplot as plt
import numpy as np
import pyNN.spiNNaker as p
from pyNN.utility.plotting import Figure, Panel

import spinn_gym as gym


rate_on = 10
rate_off = 0
pop_size = 1
prob_command = 1.0/6.0
prob_in_change = 1./2.
time_period = 200
stochastic = 0
reward = 0

p.setup(timestep=1.0)

input_size = 2
rate = 20
input_pop = p.Population(input_size, p.SpikeSourcePoisson(rate=rate))

random_seed = []
for j in range(4):
    random_seed.append(np.random.randint(0xffff))

recall_model = gym.Recall(rate_on=rate_on,
                          rate_off=rate_off,
                          pop_size=pop_size,
                          prob_command=prob_command,
                          prob_in_change=prob_in_change,
                          time_period=time_period,
                          stochastic=stochastic,
                          reward=reward,
                          random_seed=random_seed)

recall_pop = p.Population(recall_model.n_atoms, recall_model)
readout_pop = p.Population(recall_model.n_atoms, p.IF_cond_exp())

input_pop.record('spikes')
readout_pop.record('spikes')

p.external_devices.activate_live_output_to(input_pop, recall_pop)
i2o2 = p.Projection(recall_pop, readout_pop, p.OneToOneConnector(),
                    p.StaticSynapse(weight=0.1, delay=0.5))


runtime = 30 * 1000
p.run(runtime)

b_vertex = recall_pop._vertex  # pylint: disable=protected-access
scores = b_vertex.get_recorded_data('score')
scores = scores.tolist()
print(scores)

i = 0
print("score 0 \t\t|\t score 1 \t|\t\t  trials")
while i < len(scores):
    print(f"{scores[i][0]:8}\t\t{scores[i + 1][0]:8}\t\t{scores[i + 2][0]:8}")
    i += 3

accuracy = float(scores[len(scores)-2][0]+scores[len(scores)-3][0]) / float(
    scores[len(scores)-1][0])
print("Accuracy:", accuracy)

spikes_in = input_pop.get_data('spikes').segments[0].spiketrains
spikes_out = readout_pop.get_data('spikes').segments[0].spiketrains
Figure(
    Panel(spikes_in, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_out, xlabel="Time (ms)", ylabel="nID", xticks=True)
)
plt.show()

p.end()
