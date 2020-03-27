from __future__ import print_function
import spynnaker8 as p
import spinn_gym as gym
from examples.breakout.helpers import subsample_connection, paddle_and_ball_list, get_paddle_centre_projection, \
    get_paddle_lateral_connections, get_ball_x_projection, row_col_to_input_breakout
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

import subprocess
import sys

from spynnaker.pyNN.models.utility_models.spike_injector import \
    SpikeInjector
from spinn_gym.games.breakout.visualiser.visualiser import Visualiser


# -----------------------------------------------------------------------------
#  Globals
# -----------------------------------------------------------------------------
# Visualiser process (global)
vis_proc = None


# -----------------------------------------------------------------------------
#  Helper Functions
# -----------------------------------------------------------------------------

def start_visualiser(database, pop_label, xr, yr, xb=8, yb=8, key_conn=None):
    _, _, _, board_address, tag = database.get_live_output_details(
        pop_label, "LiveSpikeReceiver")

    print("Calling \'start_visualiser\'")

    # Calling visualiser - must be done as process rather than on thread due to OS security (Mac)
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


# -----------------------------------------------------------------------------
#  Initialise Simulation and Parameters
# -----------------------------------------------------------------------------

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128
PADDLE_WIDTH = 36

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

height_of_paddle = 1


# -----------------------------------------------------------------------------
#  Create Spiking Neural Network
# -----------------------------------------------------------------------------

weight_to_spike = 2

game_width = X_RESOLUTION // x_factor1
game_height = Y_RESOLUTION // y_factor1
width_of_paddle = PADDLE_WIDTH // x_factor1

b1 = gym.Breakout(x_factor=x_factor1, y_factor=y_factor1, bricking=bricking)
breakout_pop = p.Population(b1.neurons(), b1, label="breakout1")

# ex is the external device plugin manager
ex.activate_live_output_for(breakout_pop)

# Connect key spike injector to breakout population
key_input = p.Population(2, SpikeInjector, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])
p.Projection(key_input, breakout_pop, p.AllToAllConnector(), p.StaticSynapse(weight=2))

# Create random spike input and connect to Breakout pop to stimulate paddle (and enable paddle visualisation)
spike_input = p.Population(2, p.SpikeSourcePoisson(rate=10), label="input_connect")
p.Projection(spike_input, breakout_pop, p.AllToAllConnector(), p.StaticSynapse(weight=1))

weight = 0.1
[connections_on, connections_off] = subsample_connection(X_RESOLUTION/x_factor1, Y_RESOLUTION/y_factor1,
                                                         1, 1, weight, row_col_to_input_breakout)

# Create population of neurons to receive input from Breakout
receive_pop_size = game_width * game_height
receive_pop = p.Population(receive_pop_size, p.IF_cond_exp(), label="receive_pop")
p.Projection(breakout_pop, receive_pop, p.FromListConnector(connections_on), p.StaticSynapse(weight=weight_to_spike))

# Separate the receive population into ball population and paddle population
[ball_connections, paddle_connections] = paddle_and_ball_list(game_width, game_height)
paddle_pop = p.Population(game_width, p.IF_cond_exp(), label="paddle_pop")
ball_pop = p.Population(game_width * (game_height - height_of_paddle), p.IF_cond_exp(), label="ball_pop")
p.Projection(receive_pop, paddle_pop, p.FromListConnector(paddle_connections), p.StaticSynapse(weight=weight_to_spike))
p.Projection(receive_pop, ball_pop, p.FromListConnector(ball_connections), p.StaticSynapse(weight=0.1))

# Project the paddle population into the centre of paddle population
paddle_centre_pop = p.Population(game_width, p.IF_cond_exp(), label="paddle_centre_pop")
paddle_centre_list = get_paddle_centre_projection(game_width, radius=width_of_paddle//2, weight=0.1/width_of_paddle)
paddle_lateral_list = get_paddle_lateral_connections(game_width, radius=width_of_paddle//2, weight=0.1/width_of_paddle)
p.Projection(paddle_pop, paddle_centre_pop, p.FromListConnector(paddle_centre_list))
p.Projection(paddle_centre_pop, paddle_centre_pop, p.FromListConnector(paddle_lateral_list), receptor_type="inhibitory")

# Projection of the ball population onto the x-coordinate
ball_x_pop = p.Population(game_width, p.IF_cond_exp(), label="ball_x_pop")
ball_x_connections = get_ball_x_projection(game_width, game_height, weight=1)
p.Projection(ball_pop, ball_x_pop, p.FromListConnector(ball_x_connections), label="ball->ball_x")

# Hidden populations for left and right controls
hidden_left_pop = p.Population(game_width, p.IF_cond_exp(), label="hidden_left_pop")
hidden_right_pop = p.Population(game_width, p.IF_cond_exp(), label="hidden_right_pop")

# Connect centre paddle population to hidden populations
p.Projection(paddle_centre_pop, hidden_left_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=0.055), label="paddle_centre->hidden_left")
p.Projection(paddle_centre_pop, hidden_right_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=0.055), label="paddle_centre->hidden_right")

# Connect ball population to hidden populations
hidden_left_connections = []
hidden_right_connections = []

for i in range(game_width):
    for j in range(i):
        hidden_right_connections.append((i, j, 0.01, 1))
    for j in range(i + 1, game_width):
        hidden_left_connections.append((i, j, 0.01, 1))

p.Projection(ball_x_pop, hidden_left_pop,
             p.FromListConnector(hidden_left_connections), label="ball_x->hidden_left")
p.Projection(ball_x_pop, hidden_right_pop,
             p.FromListConnector(hidden_right_connections), label="ball_x->hidden_right")

# Connection lists for hidden populations to direction population (0 = left, 1 = right)
left_connections = []
right_connections = []

for i in range(game_width):
    left_connections.append((i, 0, weight, 1))
    right_connections.append((i, 1, weight, 1))

# Direction population for feeding input back into Breakout
direction_pop = p.Population(2, p.IF_cond_exp(), label="direction_pop")
p.Projection(direction_pop, breakout_pop, p.FromListConnector([(0, 1, weight, 1), (1, 2, weight, 1)]))
p.Projection(hidden_left_pop, direction_pop, p.FromListConnector(left_connections), label="hidden_left->direction")
xj = p.Projection(hidden_right_pop, direction_pop, p.FromListConnector(right_connections), label="hidden_right->direction")

# Create population to receive reward signal from Breakout (n0: rew, n1: pun)
receive_reward_pop = p.Population(2, p.IF_cond_exp(), label="receive_rew_pop")
p.Projection(breakout_pop, receive_reward_pop, p.OneToOneConnector(), p.StaticSynapse(weight=0.1*weight))

# -----------------------------------------------------------------------------
#  Setup Recording
# -----------------------------------------------------------------------------

spike_input.record('spikes')
direction_pop.record('all')
hidden_left_pop.record('spikes')
hidden_right_pop.record('spikes')
receive_pop.record('spikes')
receive_reward_pop.record('all')

ball_pop.record('spikes')
ball_x_pop.record('spikes')
paddle_pop.record('spikes')
paddle_centre_pop.record('spikes')
direction_pop.record('spikes')


# -----------------------------------------------------------------------------
#  Configure Visualiser
# -----------------------------------------------------------------------------

print("UDP_PORT1: {}".format(UDP_PORT1))
print("x_fact: {}, y_fact: {}".format(x_factor1, y_factor1))
print("x_bits: {}, y_bits: {}".format(
    np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1))),
    np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1)))
))

d_conn = DatabaseConnection(local_port=None)

print("\nRegister visualiser process")
d_conn.add_database_callback(functools.partial(
    start_visualiser, pop_label=b1.label, xr=x_factor1, yr=y_factor1,
    xb=np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1))),
    yb=np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1))),
    key_conn=key_input_connection))

p.external_devices.add_database_socket_address("localhost", d_conn.local_port, None)


# -----------------------------------------------------------------------------
#  Run Simulation
# -----------------------------------------------------------------------------

runtime = 1000 * 90
simulator = get_simulator()
print("\nLet\'s play breakout!")
p.run(runtime)


# -----------------------------------------------------------------------------
#  Post-Process Results
# -----------------------------------------------------------------------------

print("\nSimulation Complete - Extracting Data and Post-Processing")

receive_pop_spikes = receive_pop.get_data('spikes')
receive_reward_pop_output = receive_reward_pop.get_data()

left_hidden_data = hidden_left_pop.get_data('spikes')
right_hidden_data = hidden_right_pop.get_data('spikes')

ball_x_data = ball_x_pop.get_data('spikes')
paddle_centre_data = paddle_centre_pop.get_data('spikes')

direction_data = direction_pop.get_data('spikes')

print(xj.getWeights())

figure_filename = "results.png"
Figure(
    Panel(receive_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(ball_x_data.segments[0].spiketrains,
          ylabel="Ball Projection", yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(paddle_centre_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(right_hidden_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(left_hidden_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(direction_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(receive_reward_pop_output.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[receive_reward_pop.label],
          yticks=True,
          xlim=(0, runtime)
          ),
    title="Simple Breakout Example"
)

plt.show()

scores = get_scores(breakout_pop=breakout_pop, simulator=simulator)
print("Scores: {}".format(scores))

# End simulation
p.end()
vis_proc.terminate()
print("Simulation Complete")
