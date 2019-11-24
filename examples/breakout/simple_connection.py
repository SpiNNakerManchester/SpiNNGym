from __future__ import print_function

import functools
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyNN.utility.plotting import Figure, Panel

import spinn_gym as gym
import spynnaker8 as p
from spinn_front_end_common.utilities.database.database_connection \
    import DatabaseConnection
from spinn_front_end_common.utilities.globals_variables import get_simulator
from spynnaker.pyNN.connections. \
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.models.utility_models.spike_injector import \
    SpikeInjector
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex

# -----------------------------------------------------------------------------
#  Globals
# -----------------------------------------------------------------------------
vis_proc = None  # Visualiser process (global)


# -----------------------------------------------------------------------------
#  Helper Functions
# -----------------------------------------------------------------------------
def start_visualiser(database, pop_label, xr, yr, xb=8, yb=8, key_conn=None):
    _, _, _, board_address, tag = database.get_live_output_details(
        pop_label, "LiveSpikeReceiver")

    print("Calling \'start_visualiser\'")

    # Calling visualiser - must be done as process rather than on thread due to
    # OS security (Mac)
    global vis_proc
    vis_proc = subprocess.Popen(
        [sys.executable,
         '../../spinn_gym/games/breakout/visualiser/visualiser.py',
         board_address,
         tag.__str__(),
         xb.__str__(),
         yb.__str__()
         ])
    # print("Visualiser proc ID: {}".format(vis_proc.pid))


def get_scores(breakout_pop, simulator):
    b_vertex = breakout_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager,
        simulator.machine_time_step)

    return scores.tolist()


def row_col_to_input_breakout(row, col, is_on_input, row_bits, event_bits=1,
                              colour_bits=2, row_start=0):
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


def subsample_connection(x_res, y_res, subsamp_factor_x, subsamp_factor_y,
                         weight, coord_map_func):
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
            # if((j+1)%(y_res/subsamp_factor)==0 or
            # (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
            connection_list_off.append((coord_map_func(j, i, 0, row_bits),
                                        subsampidx, weight, 1.))

    return connection_list_on, connection_list_off


def separate_connections(ball_population_size, connections_on):
    pad_list = []
    ball_list = []

    for idx, val in enumerate(connections_on):
        if idx < ball_population_size:
            ball_list.append(val)
        else:
            index_in_pad_pop = idx - ball_population_size
            new_el_connection = (val[0], index_in_pad_pop, val[2], val[3])
            pad_list.append(new_el_connection)

    return ball_list, pad_list


def compress_to_x_axis(connections, x_resolution):
    compressed_connections = []

    for idx, val in enumerate(connections):
        new_el_connection = (val[0], val[1] % x_resolution, val[2], val[3])
        compressed_connections.append(new_el_connection)

    return compressed_connections


# -----------------------------------------------------------------------------
# Initialise Simulation and Parameters
# -----------------------------------------------------------------------------

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT1 = 17886
UDP_PORT2 = UDP_PORT1 + 1

# Setup pyNN simulation
p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.IF_cond_exp, 128)

# Game resolution sampling factors
x_factor1 = 2
y_factor1 = x_factor1
bricking = 1

# -----------------------------------------------------------------------------
# Create Spiking Neural Network
# -----------------------------------------------------------------------------


# Create breakout population and activate live output
b1 = gym.Breakout(x_factor=x_factor1, y_factor=y_factor1, bricking=bricking)
breakout_pop = p.Population(b1.neurons(), b1, label="breakout1")

# ex is the external device plugin manager
ex.activate_live_output_for(breakout_pop)

# Connect key spike injector to breakout population
key_input = p.Population(2, SpikeInjector, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])
p.Projection(key_input, breakout_pop, p.AllToAllConnector(),
             p.StaticSynapse(weight=0.1))

# Create random spike input and connect to Breakout pop to stimulate paddle
# (and enable paddle visualisation)
spike_input = p.Population(2, p.SpikeSourcePoisson(rate=2),
                           label="input_connect")
p.Projection(spike_input, breakout_pop, p.AllToAllConnector(),
             p.StaticSynapse(weight=0.1))

weight = 0.1
[Connections_on, Connections_off] = subsample_connection(
    X_RESOLUTION / x_factor1, Y_RESOLUTION / y_factor1, 1, 1, weight,
    row_col_to_input_breakout)

# Final Resolution
x_res = int(X_RESOLUTION / x_factor1)
y_res = int(Y_RESOLUTION / y_factor1)

# Population sizes
total_receive_pop_size = x_res * y_res

pad_pop_size = x_res
ball_pop_size = x_res

[Ball_connections, Pad_connections] = separate_connections(total_receive_pop_size - pad_pop_size, Connections_on)
Compressed_ball_connections = compress_to_x_axis(Ball_connections, x_res)

# Create the Pad position population
pad_pop = p.Population(pad_pop_size, p.IF_cond_exp(),
                       label="pad_pop")
p.Projection(breakout_pop, pad_pop, p.FromListConnector(Pad_connections),
             p.StaticSynapse(weight=weight))

# Create the Ball position population
ball_pop = p.Population(ball_pop_size, p.IF_cond_exp(),
                        label="ball_pop")
p.Projection(breakout_pop, ball_pop, p.FromListConnector(Compressed_ball_connections),
             p.StaticSynapse(weight=weight))

# Create population to receive reward signal from Breakout (n0: reward, n1: punishment)
receive_reward_pop = p.Population(2, p.IF_cond_exp(),
                                  label="receive_rew_pop")
p.Projection(breakout_pop, receive_reward_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=0.1 * weight))

# Setup recording
spike_input.record('spikes')
pad_pop.record('spikes')
ball_pop.record('spikes')
# receive_pop.record('spikes')
receive_reward_pop.record('all')

# -----------------------------------------------------------------------------
# Configure Visualiser
# -----------------------------------------------------------------------------
print("UDP_PORT1: {}".format(UDP_PORT1))
print("x_fact: {}, y_fact: {}".format(x_factor1, y_factor1))
print("x_bits: {}, y_bits: {}".format(
    np.uint32(np.ceil(np.log2(X_RESOLUTION / x_factor1))),
    np.uint32(np.ceil(np.log2(Y_RESOLUTION / y_factor1)))
))

d_conn = DatabaseConnection(local_port=None)

print("\nRegister visualiser process")
d_conn.add_database_callback(functools.partial(
    start_visualiser, pop_label=b1.label, xr=x_factor1, yr=y_factor1,
    xb=np.uint32(np.ceil(np.log2(X_RESOLUTION / x_factor1))),
    yb=np.uint32(np.ceil(np.log2(Y_RESOLUTION / y_factor1))),
    key_conn=key_input_connection))

p.external_devices.add_database_socket_address(
    "localhost", d_conn.local_port, None)

# -----------------------------------------------------------------------------
# Run Simulation
# -----------------------------------------------------------------------------
runtime = 1000 * 30
simulator = get_simulator()
print("\nLet\'s play breakout!")
p.run(runtime)

# -----------------------------------------------------------------------------
# Post-Process Results
# -----------------------------------------------------------------------------
print("\nSimulation Complete - Extracting Data and Post-Processing")

spike_input_spikes = spike_input.get_data('spikes')
pad_pop_spikes = pad_pop.get_data('spikes')
ball_pop_spikes = ball_pop.get_data('spikes')
# receive_pop_spikes = receive_pop.get_data('spikes')
receive_reward_pop_output = receive_reward_pop.get_data()

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spike_input_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(pad_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(ball_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(receive_reward_pop_output.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[receive_reward_pop.label],
          yticks=True,
          xlim=(0, runtime)
          )
    # title="Simple Breakout Example"
)

plt.show()

scores = get_scores(breakout_pop=breakout_pop, simulator=simulator)
print("Scores: {}".format(scores))

# End simulation
p.end()
vis_proc.terminate()
print("Simulation Complete")
