from __future__ import print_function

import functools
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyNN.utility.plotting import Figure, Panel

import spinn_gym as gym
import spynnaker8 as p
from examples.breakout.util import get_scores, row_col_to_input_breakout, subsample_connection, separate_connections, \
    compress_to_x_axis, generate_ball_to_hidden_pop_connections, generate_decision_connections, clean_connection
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
#  Visualiser
# ----------------------------------------------------------------------------------------------------------------------

# Visualiser process (global)
vis_proc = None


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


# ----------------------------------------------------------------------------------------------------------------------
# Initialise Simulation and Parameters
# ----------------------------------------------------------------------------------------------------------------------

# User Controls
SAVE_CONNECTIONS = False
FILENAME = "connections.json"

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

# Weights
weight = 0.1

# ----------------------------------------------------------------------------------------------------------------------
# Breakout Population && Spike Input
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
p.Projection(random_spike_input, breakout_pop, p.AllToAllConnector(), p.StaticSynapse(weight=1.))

# --------------------------------------------------------------------------------------
# ON/OFF Connections
# --------------------------------------------------------------------------------------

[Connections_on, Connections_off] = subsample_connection(X_RES, Y_RES, 1, 1, weight, row_col_to_input_breakout)

[Ball_on_connections, Paddle_on_connections] = \
    separate_connections(X_RES * Y_RES - X_RES, Connections_on)

[Ball_off_connections, Paddle_off_connections] = \
    separate_connections(X_RES * Y_RES - X_RES, Connections_off)

# --------------------------------------------------------------------------------------
# Paddle Population
# --------------------------------------------------------------------------------------

paddle_pop = p.Population(X_RES, p.IF_cond_exp(),
                          label="paddle_pop")

p.Projection(breakout_pop, paddle_pop, p.FromListConnector(Paddle_on_connections),
             receptor_type="excitatory")
p.Projection(breakout_pop, paddle_pop, p.FromListConnector(Paddle_off_connections),
             receptor_type="inhibitory")

# --------------------------------------------------------------------------------------
# Ball Position Population
# --------------------------------------------------------------------------------------

ball_pop = p.Population(X_RES, p.IF_cond_exp(),
                        label="ball_pop")

Compressed_ball_connections = compress_to_x_axis(Ball_on_connections, X_RES)

p.Projection(breakout_pop, ball_pop, p.FromListConnector(Compressed_ball_connections),
             p.StaticSynapse(weight=weight))

# --------------------------------------------------------------------------------------
# Hidden Populations
# --------------------------------------------------------------------------------------

left_hidden_pop = p.Population(X_RES, p.IF_cond_exp(),
                               label="left_hidden_pop")
right_hidden_pop = p.Population(X_RES, p.IF_cond_exp(),
                                label="right_hidden_pop")

# Project the paddle population on left/right hidden populations
# so that it charges the neurons without spiking
paddle_presence_weight = 0.01
paddle_left_projection = p.Projection(paddle_pop, left_hidden_pop, p.OneToOneConnector(),
                                      p.StaticSynapse(paddle_presence_weight))
paddle_right_projection = p.Projection(paddle_pop, right_hidden_pop, p.OneToOneConnector(),
                                       p.StaticSynapse(paddle_presence_weight))

[Ball_to_left_hidden_connections, Ball_to_right_hidden_connections] = \
    generate_ball_to_hidden_pop_connections(pop_size=X_RES, ball_presence_weight=0.07)

ball_left_projection = p.Projection(ball_pop, left_hidden_pop, p.FromListConnector(Ball_to_left_hidden_connections))
ball_right_projection = p.Projection(ball_pop, right_hidden_pop, p.FromListConnector(Ball_to_right_hidden_connections))

# --------------------------------------------------------------------------------------
# Decision Population
# --------------------------------------------------------------------------------------

decision_input_pop = p.Population(2, p.IF_cond_exp(),
                                  label="decision_input_pop")

[Left_decision_connections, Right_decision_connections] = \
    generate_decision_connections(pop_size=X_RES, decision_weight=weight)

left_decision_projection = p.Projection(left_hidden_pop, decision_input_pop,
                                        p.FromListConnector(Left_decision_connections))
right_decision_projection = p.Projection(right_hidden_pop, decision_input_pop,
                                         p.FromListConnector(Right_decision_connections))

# Connect input Decision population to the game
p.Projection(decision_input_pop, breakout_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=1.0))

# --------------------------------------------------------------------------------------
# Reward Population
# --------------------------------------------------------------------------------------

# Create population to receive reward signal from Breakout (n0: reward, n1: punishment)
receive_reward_pop = p.Population(2, p.IF_cond_exp(),
                                  label="receive_rew_pop")

p.Projection(breakout_pop, receive_reward_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=weight))

# ----------------------------------------------------------------------------------------------------------------------
# Setup recording
# ----------------------------------------------------------------------------------------------------------------------

paddle_pop.record('spikes')
ball_pop.record('spikes')
# left_hidden_pop.record('spikes')
# right_hidden_pop.record('spikes')
decision_input_pop.record('spikes')
# random_spike_input.record('spikes')
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

runtime = 1000 * 120
simulator = get_simulator()
print("\nLet\'s play breakout!")
p.run(runtime)

# ----------------------------------------------------------------------------------------------------------------------
# Post-Process Results
# ----------------------------------------------------------------------------------------------------------------------

print("\nSimulation Complete - Extracting Data and Post-Processing")

pad_pop_spikes = paddle_pop.get_data('spikes')
ball_pop_spikes = ball_pop.get_data('spikes')
# left_hidden_pop_spikes = left_hidden_pop.get_data('spikes')
# right_hidden_pop_spikes = right_hidden_pop.get_data('spikes')
decision_input_pop_spikes = decision_input_pop.get_data('spikes')
# random_spike_input_spikes = random_spike_input.get_data('spikes')
receive_reward_pop_output = receive_reward_pop.get_data()

Figure(
    Panel(pad_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(ball_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    # Panel(right_hidden_pop_spikes.segments[0].spiketrains,
    #       yticks=True, markersize=0.2, xlim=(0, runtime)),
    #
    # Panel(left_hidden_pop_spikes.segments[0].spiketrains,
    #       yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(decision_input_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    # Panel(random_spike_input_spikes.segments[0].spiketrains,
    #       yticks=True, markersize=0.2, xlim=(0, runtime)),

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

plt.figure(2)
plt.plot(scores)
plt.ylabel("score")
plt.xlabel("machine_time_step")
plt.title("Score Evolution - Automated play")

plt.show()

if SAVE_CONNECTIONS:
    print("Saving Connections")

    extracted_conn = [clean_connection(ball_left_projection.get('weight', 'list')),
                      clean_connection(paddle_left_projection.get('weight', 'list')),
                      clean_connection(ball_right_projection.get('weight', 'list')),
                      clean_connection(paddle_right_projection.get('weight', 'list')),
                      clean_connection(left_decision_projection.get('weight', 'list')),
                      clean_connection(right_decision_projection.get('weight', 'list'))]

    with open(FILENAME, "w") as f:
        f.write(json.dumps(extracted_conn))
        print("Done")


# End simulation
p.end()
vis_proc.terminate()
print("Simulation Complete")
