import spynnaker8 as p
import spinn_gym as gym
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spinn_front_end_common.utilities.globals_variables import get_simulator

import pylab
import matplotlib.pyplot as plt
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
from spynnaker import plot_utils
import threading
import time
from multiprocessing.pool import ThreadPool
import socket
import numpy as np
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# Layout of pixels
from spynnaker.pyNN.models.utility_models.spike_injector import \
    SpikeInjector
from spinn_gym.games.breakout.visualiser.visualiser import Visualiser


def thread_visualiser(UDP_PORT, xr, yr, xb=8, yb=8, key_conn=None):
    id = UDP_PORT - UDP_PORT1
    print "threadin ", running, id
    # time.sleep(5)
    # xb = np.uint32(np.ceil(np.log2(X_RESOLUTION / x_factor1)))
    # yb = np.uint32(np.ceil(np.log2(Y_RESOLUTION / y_factor1)))
    Figure
    visualiser = Visualiser(
        UDP_PORT, key_conn,# id,
        x_factor=xr, y_factor=yr,
        x_bits=xb, y_bits=yb)
    print "threadin2 ", running, id
    visualiser.show()
    # visualiser._update(None)
    # score = 0
    # while running == True:
    #     print "in ", UDP_PORT, id, score
    #     score = visualiser._update(None)
    #     time.sleep(1)
    # print "left ", running, id
    # score = visualiser._return_score()
    # visual[id] = visualiser._return_image_data()
    # result[id] = score

def get_scores(breakout_pop,simulator):
    b_vertex = breakout_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)

    return scores.tolist()

def row_col_to_input_breakout(row, col, is_on_input, row_bits, event_bits=1, colour_bits=2, row_start=0):
    row_bits = np.uint32(row_bits)
    idx = np.uint32(0)

    if is_on_input:
        idx = 1

    row += row_start
    idx = idx | (row << (colour_bits))  # colour bit
    idx = idx | (col << (row_bits + colour_bits))

    # add two to allow for special event bits
    idx = idx + 2

    return idx



def subsample_connection(x_res, y_res, subsamp_factor_x, subsamp_factor_y, weight,
                         coord_map_func):
    # subY_BITS=int(np.ceil(np.log2(y_res/subsamp_factor)))
    connection_list_on = []
    connection_list_off = []

    sx_res = int(x_res) // int(subsamp_factor_x)
    row_bits = int(np.ceil(np.log2(x_res)))
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i // subsamp_factor_x
            sj = j // subsamp_factor_y
            # ON channels
            subsampidx = sj * sx_res + si
            connection_list_on.append((coord_map_func(j, i, 1, row_bits),
                                       subsampidx, weight, 1.))
            # OFF channels only on segment borders
            # if((j+1)%(y_res/subsamp_factor)==0 or (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
            connection_list_off.append((coord_map_func(j, i, 0, row_bits),
                                        subsampidx, weight, 1.))

    return connection_list_on, connection_list_off


# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT1 = 17886
UDP_PORT2 = UDP_PORT1 + 1

# Setup pyNN simulation
p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)

x_factor1 = 2
y_factor1 = x_factor1
bricking = 0


b1 = gym.Breakout(x_factor=x_factor1, y_factor=y_factor1, bricking=bricking)

breakout_pop = p.Population(b1.neurons(), b1, label="breakout1")
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT1)

spike_input = p.Population(2, p.SpikeSourcePoisson(rate=2), label="input_connect")
p.Projection(spike_input, breakout_pop, p.AllToAllConnector(), p.StaticSynapse(weight=0.1))

# Create spike injector to inject keyboard input into simulation
key_input = p.Population(2, SpikeInjector, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])

# Connect key spike injector to breakout population
p.Projection(key_input, breakout_pop, p.AllToAllConnector(), p.StaticSynapse(weight=0.1))

weight = 0.1
[Connections_on, Connections_off]=subsample_connection(X_RESOLUTION/x_factor1, Y_RESOLUTION/y_factor1, 1, 1, weight, row_col_to_input_breakout)
receive_pop_size1 = (160/x_factor1)*(128/y_factor1)
receive_pop_1 = p.Population(receive_pop_size1, p.IF_cond_exp(), label="receive_pop")
p.Projection(breakout_pop,receive_pop_1,p.FromListConnector(Connections_on))#, p.StaticSynapse(weight=weight))


receive_pop_1.record(['spikes','gsyn_exc'])
spike_input.record('spikes')

running = True

print UDP_PORT1
print X_RESOLUTION/x_factor1
print Y_RESOLUTION/y_factor1
print np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1)))
print np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1)))
print key_input_connection


#t = threading.Thread(target=thread_visualiser, args=[UDP_PORT1, x_factor1, y_factor1,
                                                     #np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1))),
                                                     #np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1))),
                                                     #key_input_connection])

print "reached here 1"
#t.start()

runtime = 1000 * 15
simulator = get_simulator()

p.run(runtime)
print "reached here 2"

running = False

spikes_1 = receive_pop_1.get_data('spikes').segments[0].spiketrains
gsyn_exc = receive_pop_1.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]
spikes_2 = spike_input.get_data('spikes').segments[0].spiketrains
Figure(
    Panel(gsyn_exc, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_1, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_2, xlabel="Time (ms)", ylabel="nID", xticks=True)
)
plt.show()

scores = get_scores(breakout_pop=breakout_pop, simulator=simulator)

# End simulation
p.end()

print "1", scores
# print "2", scores2

