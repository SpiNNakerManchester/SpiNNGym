from __future__ import print_function

import functools
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyNN.utility.plotting import Figure, Panel

import spinn_gym as gym
import spynnaker8 as p
from examples.breakout.util import get_scores, row_col_to_input_breakout, subsample_connection, separate_connections, \
    compress_to_x_axis, generate_ball_to_hidden_pop_connections, generate_decision_connections
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

cell_params = {
    'cm': 0.3,
    'i_offset': 0.0,
    'tau_m': 10.0,
    'tau_refrac': 4.0,
    'tau_syn_E': 1.0,
    'tau_syn_I': 1.0,
    'v_reset': -70.0,
    'v_rest': -65.0,
    'v_thresh': -55.4}

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
# Reward & Punishment Population
# --------------------------------------------------------------------------------------

# n0: reward, n1: punishment
reward_conn = [(0, 0, 2, 1)]
punishment_conn = [(1, 0, 2, 1)]

reward_pop = p.Population(1, p.IF_cond_exp(),
                          label="reward_pop")
punishment_pop = p.Population(1, p.IF_cond_exp(),
                              label="punishment_pop")

p.Projection(breakout_pop, reward_pop, p.FromListConnector(reward_conn))
p.Projection(breakout_pop, punishment_pop, p.FromListConnector(punishment_conn))

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
# Hidden Populations && Neuromodulation
# --------------------------------------------------------------------------------------

# Create STDP dynamics with neuromodulation
synapse_dynamics = p.STDPMechanism(
    timing_dependence=p.IzhikevichNeuromodulation(
        tau_plus=10, tau_minus=12,
        A_plus=1, A_minus=1,
        tau_c=1000, tau_d=200),
    weight_dependence=p.MultiplicativeWeightDependence(
        w_min=0, w_max=20),
    weight=0.0,
    neuromodulation=True)

hidden_pop = p.Population(X_RES, p.IF_curr_exp_izhikevich_neuromodulation,
                          label="hidden_pop")

# Create Dopaminergic connections
reward_hidden_projection = p.Projection(
    reward_pop, hidden_pop,
    p.AllToAllConnector(),
    synapse_type=p.StaticSynapse(weight=weight),
    receptor_type='reward', label='reward synapses -> hidden')
punishment_hidden_projection = p.Projection(
    punishment_pop, hidden_pop,
    p.AllToAllConnector(),
    synapse_type=p.StaticSynapse(weight=weight),
    receptor_type='punishment', label='punishment synapses -> hidden')

# Create a plastic connection between Ball and Hidden neurons
ball_plastic_projection = p.Projection(
    ball_pop, hidden_pop,
    p.AllToAllConnector(),
    synapse_type=synapse_dynamics,
    receptor_type='excitatory', label='Ball-Hidden projection')

# Create a plastic connection between Paddle and Hidden neurons
paddle_plastic_projection = p.Projection(
    ball_pop, hidden_pop,
    p.AllToAllConnector(),
    synapse_type=synapse_dynamics,
    receptor_type='excitatory', label='Paddle-Hidden projection')

# --------------------------------------------------------------------------------------
# Decision Population && Neuromodulation
# --------------------------------------------------------------------------------------

# TODO: Stimulate the population?
decision_input_pop = p.Population(2, p.IF_curr_exp_izhikevich_neuromodulation,
                                  label="decision_input_pop")

# Create Dopaminergic connection
reward_decision_projection = p.Projection(
    reward_pop, decision_input_pop,
    p.AllToAllConnector(),
    synapse_type=p.StaticSynapse(weight=weight),
    receptor_type='reward', label='reward synapses -> decision')
punishment_decision_projection = p.Projection(
    punishment_pop, decision_input_pop,
    p.AllToAllConnector(),
    synapse_type=p.StaticSynapse(weight=weight),
    receptor_type='punishment', label='punishment synapses -> decision')

# Create a plastic connection between Hidden and Decision neurons
hidden_plastic_projection = p.Projection(
    hidden_pop, decision_input_pop,
    p.AllToAllConnector(),
    synapse_type=synapse_dynamics,
    receptor_type='excitatory', label='Hidden-Decision projection')

# Connect input Decision population to the game
p.Projection(decision_input_pop, breakout_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=1.))

# ----------------------------------------------------------------------------------------------------------------------
# Setup recording
# ----------------------------------------------------------------------------------------------------------------------

paddle_pop.record('spikes')
ball_pop.record('spikes')
decision_input_pop.record('spikes')
random_spike_input.record('spikes')
reward_pop.record('all')
punishment_pop.record('all')

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

runtime = 1000 * 60
simulator = get_simulator()
print("\nLet\'s play breakout!")
p.run(runtime)

# ----------------------------------------------------------------------------------------------------------------------
# Post-Process Results
# ----------------------------------------------------------------------------------------------------------------------

print("\nSimulation Complete - Extracting Data and Post-Processing")

pad_pop_spikes = paddle_pop.get_data('spikes')
ball_pop_spikes = ball_pop.get_data('spikes')
decision_input_pop_spikes = decision_input_pop.get_data('spikes')
random_spike_input_spikes = random_spike_input.get_data('spikes')
reward_pop_output = reward_pop.get_data()
punishment_pop_output = punishment_pop.get_data()

Figure(
    Panel(pad_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(ball_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(decision_input_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(random_spike_input_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(reward_pop_output.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[reward_pop.label],
          yticks=True,
          xlim=(0, runtime)
          ),
    Panel(punishment_pop_output.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[punishment_pop.label],
          yticks=True,
          xlim=(0, runtime)
          )
    # title="Simple Breakout Example"
)

plt.show()

scores = get_scores(breakout_pop=breakout_pop, simulator=simulator)
print("Scores: {}".format(scores))

# TODO: methods to save and load weights
ball_weights = ball_plastic_projection.get('weight', 'list')
paddle_weights = paddle_plastic_projection.get('weight', 'list')
hidden_weights = hidden_plastic_projection.get('weight', 'list')

print("Ball -> Hidden weights: " + repr(ball_weights))
print("Paddle -> Hidden weights: " + repr(paddle_weights))
print("Hidden -> Decision weights: " + repr(hidden_weights))

plt.figure(2)
plt.plot(scores)
plt.ylabel("score")
plt.xlabel("machine_time_step")
plt.title("Score Evolution - Automated play")

plt.show()

# End simulation
p.end()
vis_proc.terminate()
print("Simulation Complete")
