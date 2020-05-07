import spynnaker8 as p
from spynnaker.pyNN.connections. \
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
import spinn_gym as gym

import pylab
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from spinn_front_end_common.utilities.globals_variables import get_simulator


def get_scores(recall_pop, simulator):
    b_vertex = recall_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)

    return scores.tolist()

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
                          rand_seed=random_seed)

recall_pop = p.Population(recall_model.neurons(), recall_model)

readout_pop = p.Population(recall_model.neurons(), p.IF_cond_exp())

input_pop.record('spikes')
# recall_pop.record('spikes')
readout_pop.record('spikes')

i2a = p.Projection(input_pop, recall_pop, p.AllToAllConnector())

# test_rec = p.Projection(recall_pop, recall_pop, p.AllToAllConnector(), p.StaticSynapse(weight=0.1, delay=0.5))
i2o2 = p.Projection(recall_pop, readout_pop, p.OneToOneConnector(), p.StaticSynapse(weight=0.1, delay=0.5))

simulator = get_simulator()

runtime = 30 * 1000
p.run(runtime)

scores = get_scores(recall_pop=recall_pop, simulator=simulator)

print(scores)

i = 0
print("score  \t\t|\t\t  trials")
while i < len(scores):
    print("{:8}\t{:8}".format(scores[i][0], scores[i+1][0]))
    i += 2

accuracy = float(scores[len(scores)-2][0]) / float(scores[len(scores)-1][0])
print("Accuracy:", accuracy)

spikes_in = input_pop.get_data('spikes').segments[0].spiketrains
spikes_out = readout_pop.get_data('spikes').segments[0].spiketrains
Figure(
    Panel(spikes_in, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_out, xlabel="Time (ms)", ylabel="nID", xticks=True)
)
plt.show()

p.end()





