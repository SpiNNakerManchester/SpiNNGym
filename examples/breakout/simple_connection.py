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

# ----------------------------------------------------------------------------------------------------------------------
#  Globals
# ----------------------------------------------------------------------------------------------------------------------
vis_proc = None  # Visualiser process (global)


# ----------------------------------------------------------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------------------------------------------------------
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


# Separates the ball and pad connections in different populations
def separate_connections(ball_population_size, connections_on):
    paddle_list = []
    ball_list = []

    for idx, val in enumerate(connections_on):
        if idx < ball_population_size:
            ball_list.append(val)
        else:
            index_in_paddle_pop = idx - ball_population_size
            new_el_connection = (val[0], index_in_paddle_pop, val[2], val[3])
            paddle_list.append(new_el_connection)

    return ball_list, paddle_list


# Get connections of compressed PADDLE population to one neuron each
def map_to_one_neuron_per_paddle(pop_size, no_paddle_neurons, syn_weight, paddle_connections):
    connections = []

    no_paddle_neurons = int(no_paddle_neurons)
    offset = no_paddle_neurons // 2

    for idx, val in enumerate(paddle_connections):
        from_neuron = val[1] - offset
        to_neuron = val[1] + offset + 1

        for neuron_no in range(from_neuron, to_neuron):
            if 0 <= neuron_no < pop_size:
                connections.append((val[0], neuron_no, syn_weight, val[3]))

    return connections


def create_lateral_inhibitory_paddle_connections(pop_size, no_paddle_neurons, syn_weight):
    lat_connections = []

    no_paddle_neurons = int(no_paddle_neurons)
    # just a precaution
    no_paddle_neurons += 2
    paddle_neurons_offset = no_paddle_neurons // 2

    # If the no_pad_neurons is even
    # then recalculate the offset
    if no_paddle_neurons % 2 == 0:
        paddle_neurons_offset -= 1

    for neuron in range(0, pop_size):
        for paddle_neuron in range(neuron - paddle_neurons_offset, neuron + paddle_neurons_offset + 1):
            if paddle_neuron != neuron and 0 <= paddle_neuron < pop_size:
                # I used to calculate the weight based on the number of excitatory input connections
                new_weight = syn_weight * (no_paddle_neurons - abs(neuron - paddle_neuron))
                lat_connections.append((neuron, paddle_neuron, new_weight, 1.))

    return lat_connections


# Get connections of compressed BALL population to the X axis
def compress_to_x_axis(connections, x_resolution):
    compressed_connections = []

    for idx, val in enumerate(connections):
        new_el_connection = (val[0], val[1] % x_resolution, val[2], val[3])
        compressed_connections.append(new_el_connection)

    return compressed_connections


def generate_ball_to_hidden_pop_connections(pop_size, ball_presence_weight=0.05):
    left_connections = []
    right_connections = []

    for ball_neuron in range(0, pop_size):
        # Connect the ball neuron to all the neurons to the left of it in the left hidden population
        for left_hidden_neuron in range(0, ball_neuron):
            right_connections.append((ball_neuron, left_hidden_neuron, ball_presence_weight, 1.))
        # Connect the ball neuron to all the neurons to the right of it in the right hidden population
        for right_hidden_neuron in range(ball_neuron + 1, pop_size):
            left_connections.append((ball_neuron, right_hidden_neuron, ball_presence_weight, 1.))

    return left_connections, right_connections


def generate_decision_connections(pop_size, decision_weight):
    left_conn = []
    right_conn = []

    for neuron in range(0, pop_size):
        left_conn.append((neuron, 0, decision_weight, 1.))
        right_conn.append((neuron, 1, decision_weight, 1.))

    return left_conn, right_conn


# ----------------------------------------------------------------------------------------------------------------------
# Initialise Simulation and Parameters
# ----------------------------------------------------------------------------------------------------------------------


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

# Final Resolution
X_RES = int(X_RESOLUTION / x_factor1)
Y_RES = int(Y_RESOLUTION / y_factor1)

# Population sizes
paddle_pop_size = X_RES
ball_pop_size = X_RES
hidden_pop_size = X_RES
breakout_pop_size = X_RES * Y_RES

# Weights
weight = 0.1

# based on the size of the bat in bkout.c --> pad_neuron_size =  bat_len // 2
paddle_neuron_size = 50 // 2

# ----------------------------------------------------------------------------------------------------------------------
# Create Spiking Neural Network
# ----------------------------------------------------------------------------------------------------------------------


# Create breakout population and activate live output
b1 = gym.Breakout(x_factor=x_factor1, y_factor=y_factor1, bricking=bricking)
breakout_pop = p.Population(b1.neurons(), b1, label="breakout1")

# ex is the external device plugin manager
ex.activate_live_output_for(breakout_pop)

key_input = p.Population(2, SpikeInjector, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])


# Create random spike input and connect to Breakout pop to stimulate paddle
# (and enable paddle visualisation)
random_spike_input = p.Population(2, p.SpikeSourcePoisson(rate=7),
                                  label="input_connect")
p.Projection(random_spike_input, breakout_pop, p.OneToOneConnector(), p.StaticSynapse(weight=1.))

[Connections_on, Connections_off] = subsample_connection(
    X_RESOLUTION / x_factor1, Y_RESOLUTION / y_factor1, 1, 1, weight,
    row_col_to_input_breakout)

[Ball_connections, Paddle_connections] = separate_connections(breakout_pop_size - paddle_pop_size, Connections_on)

# Calculated using pad_neuron_size * weight = 5 // to fire
# and (pad_neuron_size - 1) * weight <= 4.75 // to not fire
# Triggers only the middle neuron of the pad
# paddle_to_one_neuron_weight = 0.0035 - For 25 neurons per paddle
paddle_to_one_neuron_weight = 0.0875 / paddle_neuron_size
Compressed_paddle_connections = map_to_one_neuron_per_paddle(paddle_pop_size, paddle_neuron_size,
                                                             paddle_to_one_neuron_weight, Paddle_connections)

Inhibitory_lateral_paddle_connections = create_lateral_inhibitory_paddle_connections(paddle_pop_size,
                                                                                     paddle_neuron_size,
                                                                                     paddle_to_one_neuron_weight * 4)

Compressed_ball_connections = compress_to_x_axis(Ball_connections, X_RES)

# Create the Pad position population
paddle_pop = p.Population(paddle_pop_size, p.IF_cond_exp(),
                          label="paddle_pop")
p.Projection(breakout_pop, paddle_pop, p.FromListConnector(Compressed_paddle_connections),
             p.StaticSynapse(weight=paddle_to_one_neuron_weight, delay=1.))

# If a neuron fired then discharge all the other charged neurons
p.Projection(paddle_pop, paddle_pop, p.FromListConnector(Inhibitory_lateral_paddle_connections),
             synapse_type=p.StaticSynapse(weight=-paddle_to_one_neuron_weight, delay=1.), receptor_type='inhibitory')

# Create the Ball position population
ball_pop = p.Population(ball_pop_size, p.IF_cond_exp(),
                        label="ball_pop")
p.Projection(breakout_pop, ball_pop, p.FromListConnector(Compressed_ball_connections),
             p.StaticSynapse(weight=weight))

# Create the hidden populations
left_hidden_pop = p.Population(hidden_pop_size, p.IF_cond_exp(),
                               label="left_hidden_pop")
right_hidden_pop = p.Population(hidden_pop_size, p.IF_cond_exp(),
                                label="right_hidden_pop")

# Project the paddle population on left/right hidden populations
# so that it charges the neurons without spiking
p.Projection(paddle_pop, left_hidden_pop, p.OneToOneConnector(),
             p.StaticSynapse(0.05))
p.Projection(paddle_pop, right_hidden_pop, p.OneToOneConnector(),
             p.StaticSynapse(0.05))

[Ball_to_left_hidden_connections, Ball_to_right_hidden_connections] = \
    generate_ball_to_hidden_pop_connections(pop_size=X_RES, ball_presence_weight=0.07)

p.Projection(ball_pop, left_hidden_pop, p.FromListConnector(Ball_to_left_hidden_connections))
p.Projection(ball_pop, right_hidden_pop, p.FromListConnector(Ball_to_right_hidden_connections))

[Left_decision_connections, Right_decision_connections] = \
    generate_decision_connections(pop_size=X_RES, decision_weight=0.1)

# Create the decision population
decision_input_pop = p.Population(2, p.IF_cond_exp(),
                                  label="decision_input_pop")
p.Projection(left_hidden_pop, decision_input_pop, p.FromListConnector(Left_decision_connections))
p.Projection(right_hidden_pop, decision_input_pop, p.FromListConnector(Right_decision_connections))

# Connect input Decision population to the game
p.Projection(decision_input_pop, breakout_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=1.0))


# Create population to receive reward signal from Breakout (n0: reward, n1: punishment)
receive_reward_pop = p.Population(2, p.IF_cond_exp(),
                                  label="receive_rew_pop")
p.Projection(breakout_pop, receive_reward_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=0.1 * weight))

# Setup recording
paddle_pop.record('spikes')
ball_pop.record('spikes')
left_hidden_pop.record('spikes')
right_hidden_pop.record('spikes')
decision_input_pop.record('spikes')
# spike_input.record('spikes')
receive_reward_pop.record('all')

# ----------------------------------------------------------------------------------------------------------------------
# Configure Visualiser
# ----------------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------
# Run Simulation
# ----------------------------------------------------------------------------------------------------------------------

runtime = 1000 * 15
simulator = get_simulator()
print("\nLet\'s play breakout!")
p.run(runtime)

# ----------------------------------------------------------------------------------------------------------------------
# Post-Process Results
# ----------------------------------------------------------------------------------------------------------------------

print("\nSimulation Complete - Extracting Data and Post-Processing")

pad_pop_spikes = paddle_pop.get_data('spikes')
ball_pop_spikes = ball_pop.get_data('spikes')
left_hidden_pop_spikes = left_hidden_pop.get_data('spikes')
right_hidden_pop_spikes = right_hidden_pop.get_data('spikes')
decision_input_pop_spikes = decision_input_pop.get_data('spikes')
# spike_input_spikes = spike_input.get_data('spikes')
receive_reward_pop_output = receive_reward_pop.get_data()

Figure(
    Panel(pad_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(ball_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(left_hidden_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(right_hidden_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(decision_input_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    # Panel(spike_input_spikes.segments[0].spiketrains,
    #       yticks=True, markersize=0.2, xlim=(0, runtime)),

    # Panel(receive_reward_pop_output.segments[0].filter(name='gsyn_exc')[0],
    #       ylabel="gsyn excitatory (mV)",
    #       data_labels=[receive_reward_pop.label],
    #       yticks=True,
    #       xlim=(0, runtime)
    #       )
    # title="Simple Breakout Example"
)

plt.show()

scores = get_scores(breakout_pop=breakout_pop, simulator=simulator)
print("Scores: {}".format(scores))

# End simulation
p.end()
vis_proc.terminate()
print("Simulation Complete")
