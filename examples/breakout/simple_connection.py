import spynnaker8 as p
import spinn_gym as gym
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spinn_front_end_common.utilities.globals_variables import get_simulator
from spinn_front_end_common.utilities.database.database_connection \
    import DatabaseConnection

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
import functools

# Layout of pixels
from spynnaker.pyNN.models.utility_models.spike_injector import \
    SpikeInjector
from spinn_gym.games.breakout.visualiser.visualiser import Visualiser


def thread_visualiser(board_address, tag, xr, yr, xb, yb, key_conn):
    # time.sleep(5)
    # xb = np.uint32(np.ceil(np.log2(X_RESOLUTION / x_factor1)))
    # yb = np.uint32(np.ceil(np.log2(Y_RESOLUTION / y_factor1)))
    visualiser = Visualiser(
        board_address, tag, key_conn,
        x_factor=xr, y_factor=yr,
        x_bits=xb, y_bits=yb)
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


def start_visualiser(database, pop_label, xr, yr, xb=8, yb=8, key_conn=None):
    _, _, _, board_address, tag = database.get_live_output_details(
        pop_label, "LiveSpikeReceiver")
    thread = threading.Thread(target=thread_visualiser, args=[
        board_address, tag, xr, yr, xb, yb, key_conn])
    thread.start()


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
    row_bits = int(np.ceil(np.log2(y_res)))
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
bricking = 1

# Create breakout population and activate live output for it
# breakout_pop = p.Population(1, p.Breakout(WIDTH_PIXELS=(X_RESOLUTION/x_factor1), HEIGHT_PIXELS=(Y_RESOLUTION/y_factor1), label="breakout1"))
# breakout_pop2 = p.Population(1, p.Breakout(WIDTH_PIXELS=(X_RESOLUTION/x_factor2), HEIGHT_PIXELS=(Y_RESOLUTION/y_factor2), label="breakout2"))

b1 = gym.Breakout(x_factor=x_factor1, y_factor=y_factor1, bricking=bricking)

breakout_pop = p.Population(b1.neurons(), b1, label="breakout1")
# b2 = b_out(x_factor=x_factor2, y_factor=y_factor2, bricking=0)
# breakout_pop2 = p.Population(b2.neurons(), b2, label="breakout2")


ex.activate_live_output_for(breakout_pop)


# ex.activate_live_output_for(breakout_pop2, host="0.0.0.0", port=UDP_PORT2)


# Connect key spike injector to breakout population
# rate = {'rate': 2}#, 'duration': 10000000}
# spike_input = p.Population(2, p.SpikeSourcePoisson(rate=2), label="input_connect")
# p.Projection(spike_input, breakout_pop, p.AllToAllConnector(), p.StaticSynapse(weight=0.1))
# key_input_connection = None

# Create spike injector to inject keyboard input into simulation
key_input = p.Population(2, SpikeInjector, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])

# Connect key spike injector to breakout population
p.Projection(key_input, breakout_pop, p.AllToAllConnector(), p.StaticSynapse(weight=0.1))

# spike_input2 = p.Population(2, p.SpikeSourcePoisson(rate=2), label="input_connect")
# p.Projection(spike_input2, breakout_pop2, p.AllToAllConnector(), p.StaticSynapse(weight=0.1))
# key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["input_connect"])

weight = 0.1
[Connections_on, Connections_off]=subsample_connection(X_RESOLUTION/x_factor1, Y_RESOLUTION/y_factor1, 1, 1, weight, row_col_to_input_breakout)
# receive_pop_size1 = (160/x_factor1)*(128/y_factor1)
# receive_pop_size2 = (160/x_factor2)*(128/y_factor2)
# receive_pop_1 = p.Population(receive_pop_size1, p.IF_cond_exp(), label="receive_pop")
# receive_pop_2 = p.Population(receive_pop_size2, p.IF_cond_exp(), label="receive_pop")
# p.Projection(breakout_pop,receive_pop_1,p.FromListConnector(Connections_on))#, p.StaticSynapse(weight=weight))
# p.Projection(breakout_pop2,receive_pop_2,p.OneToOneConnector(), p.StaticSynapse(weight=weight))
# receive_pop_1.record('spikes')#["spikes"])
# receive_pop_2.record('spikes')#["spikes"])

test_pop = p.Population(1, p.IF_cond_exp(), label="test_pop")
p.Projection(breakout_pop, test_pop, p.OneToOneConnector(), p.StaticSynapse(weight=weight))
# test_pop.record('spikes')

# Create visualiser
# visualiser = Visualiser(
#     UDP_PORT1, None,
#     x_res=X_RESOLUTION/x_factor1, y_res=Y_RESOLUTION/y_factor1,
#     x_bits=np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1))), y_bits=np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1))))

running = True
# t = threading.Thread(target=thread_visualiser, args=[UDP_PORT1, X_RESOLUTION/x_factor1, Y_RESOLUTION/y_factor1])

print UDP_PORT1
print X_RESOLUTION/x_factor1
print Y_RESOLUTION/y_factor1
print np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1)))
print np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1)))
print key_input_connection

# visualiser = Visualiser(
#         UDP_PORT, key_conn,# id,
#         x_res=xr, y_res=yr,
#         x_bits=xb, y_bits=yb)


# visualiser = Visualiser(UDP_PORT1, key_input_connection,
#                         X_RESOLUTION/x_factor1, Y_RESOLUTION/y_factor1,
#                         np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1))),
#                         np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1)))
#                         )


# t = threading.Thread(target=thread_visualiser, args=[UDP_PORT2, X_RESOLUTION/x_factor1, Y_RESOLUTION/y_factor1,
#                                                      np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1)))-1,
#                                                      np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1)))-1])
# r = threading.Thread(target=thread_visualiser, args=[UDP_PORT2])
# result = [10 for i in range(2)]
# x_res=160
# y_res=128
# visual = [np.zeros((y_res, x_res)) for i in range(2)]
# t = ThreadPool(processes=2)
# r = ThreadPool(processes=2)
# result = t.apply_async(thread_visualiser, [UDP_PORT1])
# result2 = r.apply_async(thread_visualiser, [UDP_PORT2])
# t.daemon = True
# Run simulation (non-blocking)
# visualiser.show()

d_conn = DatabaseConnection(local_port=None)
d_conn.add_database_callback(functools.partial(
    start_visualiser, pop_label=b1.label, xr=x_factor1, yr=y_factor1,
    xb=np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1))),
    yb=np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1))),
    key_conn=key_input_connection))
p.external_devices.add_database_socket_address(
     "localhost", d_conn.local_port, None)


print "reached here 1"
# r.start()
runtime = 1000 * 30

simulator = get_simulator()



p.run(runtime)
print "reached here 2"

# visualiser.show()

running = False
# visualiser._return_score()

# Show visualiser (blocking)
# visualiser.show()

# for j in range(receive_pop_size):
# spikes_1 = receive_pop_1.get_data('spikes').segments[0].spiketrains

# counter = 0
# for neuron in spikes_1:
#     for spike in neuron:
#         print spike
#         counter += 1
# spikes_2 = receive_pop_2.get_data('spikes').segments[0].spiketrains
# spikes_t = test_pop.get_data('spikes').segments[0].spiketrains
# Figure(
#     Panel(spikes_1, xlabel="Time (ms)", ylabel="nID", xticks=True)#,
#     # Panel(spikes_2, xlabel="Time (ms)", ylabel="nID", xticks=True)#,
#     # Panel(spikes_t, xlabel="Time (ms)", ylabel="nID", xticks=True)
# )
# plt.show()

scores = get_scores(breakout_pop=breakout_pop, simulator=simulator)
# scores2 = get_scores(breakout_pop=breakout_pop2, simulator=simulator)

# End simulation
p.end()

print "1", scores
# print "2", scores2

