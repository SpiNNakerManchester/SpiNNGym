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
import numpy as np
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import spinn_gym as gym
# from spinn_arm.python_models.arm import Arm
from spinn_front_end_common.data import FecDataView


def get_scores(game_pop):
    g_vertex = game_pop._vertex
    scores = g_vertex.get_data('score')
    return scores.tolist()


def connect_to_arms(pre_pop, from_list, arms, r_type, plastic, stdp_model):
    arm_conn_list = []
    for i in range(len(arms)):
        arm_conn_list.append([])
    for conn in from_list:
        arm_conn_list[conn[1]].append((conn[0], 0, conn[2], conn[3]))
        # print "out:", conn[1]
        # if conn[1] == 2:
        #     print '\nit is possible\n'
    for i in range(len(arms)):
        if len(arm_conn_list[i]) != 0:
            if plastic:
                p.Projection(pre_pop, arms[i],
                             p.FromListConnector(arm_conn_list[i]),
                             receptor_type=r_type, synapse_type=stdp_model)
            else:
                p.Projection(pre_pop, arms[i],
                             p.FromListConnector(arm_conn_list[i]),
                             receptor_type=r_type)


runtime = 21000
exposure_time = 200
encoding = 1
time_increment = 20
pole_length = 1
pole2_length = 0.1
pole_angle = 0.1
pole2_angle = 0
reward_based = 0
force_increments = 20
max_firing_rate = 1000
number_of_bins = 6
central = 1
bin_overlap = 2

inputs = 2
outputs = 2

p.setup(timestep=1.0, min_delay=1)
p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
# One of these variables can be replaced with test_data
# depending on what needs to be tested
input_model = gym.DoublePendulum(
    encoding=encoding, time_increment=time_increment, pole_length=pole_length,
    pole2_length=pole2_length, pole_angle=pole_angle, pole2_angle=pole2_angle,
    reward_based=reward_based, force_increments=force_increments,
    max_firing_rate=max_firing_rate, number_of_bins=number_of_bins,
    central=central, rand_seed=[np.random.randint(0xffff) for i in range(4)],
    bin_overlap=3, label='pendulum_pop')

pendulum_pop_size = input_model.neurons()
pendulum = p.Population(pendulum_pop_size, input_model)
null_pop = p.Population(6*number_of_bins, p.IF_cond_exp(), label='null')
p.Projection(pendulum, null_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=0.09))
null_pop.record(['spikes', 'v', 'gsyn_exc'])
# null_pop.record(['spikes', 'v'])
# null_pops = []
# for i in range(4*number_of_bins):
#     null_pops.append(p.Population(1, p.IF_cond_exp(),
#                                   label='null {}'.format(i)))
#     null_pops[i].record(['spikes', 'v'])
#     p.Projection(pendulum, null_pops[i],
#                  p.FromListConnector([[i, 0, weight, 1]]))

arm_collection = []
input_spikes = []
rates = [0, 0]
# rates = [0, 20]
# rates = [0, 10]
print('rates = ', rates)
weight = 0.1
label = '64 0.55'
from_list_conn_left = [[0, 0, weight, 1], [6, 0, weight, 1],
                       [3, 0, weight, 1], [11, 0, weight, 1]]
from_list_conn_right = [[2, 0, weight, 1], [8, 0, weight, 1],
                        [5, 0, weight, 1], [9, 0, weight, 1]]
left = 0
right = 1
from_list_conn_out = [[0, left, weight, 1], [6, left, weight, 1],
                      [3, left, weight, 1], [11, left, weight, 1],
                      [2, right, weight, 1], [8, right, weight, 1],
                      [5, right, weight, 1], [9, right, weight, 1]]
output_pop = p.Population(
    outputs, p.IF_cond_exp(
        tau_m=0.5, tau_refrac=0, v_thresh=-64, tau_syn_E=1, tau_syn_I=1),
    label='out')
output_pop.record(['spikes', 'v', 'gsyn_exc'])
p.Projection(pendulum, output_pop, p.FromListConnector(from_list_conn_out))
output_pop2 = p.Population(
    outputs, p.IF_cond_exp(
        tau_m=0.5, tau_refrac=0, v_thresh=-64, tau_syn_E=0.5, tau_syn_I=0.5),
    label='out2')
output_pop2.record(['spikes', 'v', 'gsyn_exc'])
p.Projection(null_pop, output_pop2, p.FromListConnector(from_list_conn_out))
arm_conns = [from_list_conn_left, from_list_conn_right]
# for j in range(outputs):
#     arm_collection.append(p.Population(
#         int(np.ceil(np.log2(outputs))),
#         Arm(arm_id=j, reward_delay=exposure_time,
#             rand_seed=[np.random.randint(0xffff) for k in range(4)],
#             no_arms=outputs, arm_prob=1),
#         label='arm_pop{}'.format(j)))
#     p.Projection(arm_collection[j], pendulum, p.AllToAllConnector(),
#                  p.StaticSynapse())
#     p.Projection(null_pop, arm_collection[j], p.AllToAllConnector())
#     input_spikes.append(p.Population(1, p.SpikeSourcePoisson(rate=rates[j])))
#     p.Projection(input_spikes[j], arm_collection[j], p.AllToAllConnector(),
#                  p.StaticSynapse())
#     p.Projection(null_pop, arm_collection[j],
#                  p.FromListConnector(arm_conns[j]))
# for conn in from_list_conn_left:
#     p.Projection(null_pops[conn[0]], arm_collection[0],
#                  p.AllToAllConnector())
# for conn in from_list_conn_right:
#     p.Projection(null_pops[conn[0]], arm_collection[1],
#                  p.AllToAllConnector())

p.run(runtime)

scores = []
scores.append(get_scores(game_pop=pendulum))
if reward_based:
    print(scores)
else:
    i = 0
    print("cart  |  angle")
    while i < len(scores[0]):
        print("{}\t{}\t{}".format(
            scores[0][i], scores[0][i+1], scores[0][i+2]))
        i += 3

# spikes = []
# v = []
# for i in range(4*number_of_bins):
#     spikes.append(null_pops[i].get_data('spikes').segments[0].spiketrains)
#     v.append(null_pops[i].get_data('v').segments[0].filter(name='v')[0])
# plt.figure("spikes out")
# Figure(
#         Panel(spikes[0], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[1], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[2], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[3], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[4], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[5], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[6], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[7], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[8], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[9], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[10], xlabel="Time (ms)", ylabel="nID", xticks=True),
#         Panel(spikes[11], xlabel="Time (ms)", ylabel="nID", xticks=True)
# )
# plt.show()

spikes_n = null_pop.get_data('spikes').segments[0].spiketrains
v_n = null_pop.get_data('v').segments[0].filter(name='v')[0]
spikes_o = output_pop.get_data('spikes').segments[0].spiketrains
v_o = output_pop.get_data('v').segments[0].filter(name='v')[0]
g_o = output_pop.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]
spikes_o2 = output_pop2.get_data('spikes').segments[0].spiketrains
v_o2 = output_pop2.get_data('v').segments[0].filter(name='v')[0]
g_o2 = output_pop2.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]
plt.figure("spikes out {}".format(label))
Figure(
    Panel(spikes_n, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(v_n, ylabel="Membrane potential (mV)", yticks=True),
    Panel(spikes_o, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(v_o, ylabel="Membrane potential (mV)", yticks=True),
    Panel(g_o, ylabel="Gsyn_exc potential (mV)", yticks=True),
    Panel(spikes_o2, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(v_o2, ylabel="Membrane potential (mV)", yticks=True),
    Panel(g_o2, ylabel="Gsyn_exc potential (mV)", yticks=True)
)
plt.show()

print('rates = ', rates)
p.end()
