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

import pyNN.spiNNaker as p
import spinn_gym as gym

from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np


def get_scores(logic_pop):
    b_vertex = logic_pop._vertex
    scores = b_vertex.get_data('score')
    return scores.tolist()


p.setup(timestep=1.0)

truth_table = [0, 1, 1, 0]
input_sequence = [1, 1]

# index 0 = off
# index 1 = on
model_input = [0, 1]

input_size = len(input_sequence)
rate = 20
input_pop = p.Population(input_size, p.SpikeSourcePoisson(
    rate=[rate*model_input[0], rate*model_input[1]]))

output_pop1 = p.Population(2, p.IF_cond_exp())
output_pop2 = p.Population(2, p.IF_cond_exp())

random_seed = []
for j in range(4):
    random_seed.append(np.random.randint(0xffff))

logic_model = gym.Logic(truth_table=truth_table,
                        input_sequence=input_sequence,
                        stochastic=0,
                        rand_seed=random_seed)
logic_pop = p.Population(input_size, logic_model)

input_pop.record('spikes')
# logic_pop.record('spikes')
output_pop1.record('spikes')
output_pop2.record('spikes')

i2a = p.Projection(input_pop, logic_pop, p.AllToAllConnector())

# test_rec = p.Projection(logic_pop, logic_pop, p.AllToAllConnector(),
#                     p.StaticSynapse(weight=0.1, delay=0.5))
i2o1 = p.Projection(logic_pop, output_pop1, p.AllToAllConnector(),
                    p.StaticSynapse(weight=0.1, delay=1))
i2o2 = p.Projection(logic_pop, output_pop2, p.OneToOneConnector(),
                    p.StaticSynapse(weight=0.1, delay=1))

runtime = 10000
p.run(runtime)

scores = get_scores(logic_pop=logic_pop)

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
