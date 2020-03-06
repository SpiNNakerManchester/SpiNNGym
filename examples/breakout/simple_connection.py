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
    compress_to_x_axis, get_hidden_to_decision_connections, clean_connection, map_to_one_neuron_per_paddle, \
    create_lateral_inhibitory_paddle_connections
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

# User controls
TESTING_TIME = 1000 * 20
TRAINING_TIME = 1000 * 60 * 5
SIMULATION_TIME = TRAINING_TIME if sys.argv[1] == "Training" else TESTING_TIME

RANDOM_SPIKE_INPUT = False
LOAD_PREVIOUS_CONNECTIONS = True
SAVE_CONNECTIONS = True if sys.argv[1] == "Training" else False
TESTING = True if sys.argv[1] == "Testing" else False
FILENAME = 'connections.json'

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT1 = 17886
UDP_PORT2 = UDP_PORT1 + 1

# Setup pyNN simulation
p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.IF_cond_exp, 64)
p.set_number_of_neurons_per_core(p.IF_curr_exp_izhikevich_neuromodulation, 32)

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
# Load previous data
# ----------------------------------------------------------------------------------------------------------------------

if LOAD_PREVIOUS_CONNECTIONS:
    with open(FILENAME, "r") as f:
        previous_connections = json.loads(f.read())

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
if RANDOM_SPIKE_INPUT:
    random_spike_input = p.Population(2, p.SpikeSourcePoisson(rate=2),
                                      label="input_connect")
    p.Projection(random_spike_input, breakout_pop, p.AllToAllConnector(), p.StaticSynapse(weight=1.))

# --------------------------------------------------------------------------------------
# Reward & Punishment Population
# --------------------------------------------------------------------------------------

# n0: reward, n1: punishment
reward_conn = [(0, 0, 2, 1)]
punishment_ball_on_left_conn = [(1, 0, 2, 1)]
punishment_ball_on_right_conn = [(2, 0, 2, 1)]

reward_pop = p.Population(1, p.IF_cond_exp(),
                          label="reward_pop")
ball_on_left_dopaminergic_pop = p.Population(1, p.IF_cond_exp(),
                                             label="punishment_pop")
ball_on_right_dopaminergic_pop = p.Population(1, p.IF_cond_exp(),
                                              label="new_dopaminergic_pop")

p.Projection(breakout_pop, reward_pop, p.FromListConnector(reward_conn))
p.Projection(breakout_pop, ball_on_left_dopaminergic_pop, p.FromListConnector(punishment_ball_on_left_conn))
p.Projection(breakout_pop, ball_on_right_dopaminergic_pop, p.FromListConnector(punishment_ball_on_right_conn))

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

# based on the size of the bat in bkout.c
paddle_neuron_size = 30 // x_factor1
paddle_to_one_neuron_weight = 0.0875 / paddle_neuron_size

Compressed_paddle_connections = map_to_one_neuron_per_paddle(X_RES, paddle_neuron_size, paddle_to_one_neuron_weight,
                                                             Paddle_on_connections)
Lat_inh_connections = create_lateral_inhibitory_paddle_connections(X_RES, paddle_neuron_size,
                                                                   paddle_to_one_neuron_weight)

paddle_pop = p.Population(X_RES, p.IF_cond_exp(),
                          label="paddle_pop")
p.Projection(breakout_pop, paddle_pop, p.FromListConnector(Compressed_paddle_connections),
             receptor_type="excitatory")
p.Projection(paddle_pop, paddle_pop, p.FromListConnector(Lat_inh_connections),
             receptor_type="inhibitory")

# --------------------------------------------------------------------------------------
# Ball Positions Populations
# --------------------------------------------------------------------------------------

ball_x_pop = p.Population(X_RES, p.IF_cond_exp(),
                          label="ball_x_pop")

Ball_x_connections = compress_to_x_axis(Ball_on_connections, X_RES)

p.Projection(breakout_pop, ball_x_pop, p.FromListConnector(Ball_x_connections),
             p.StaticSynapse(weight=weight))

# --------------------------------------------------------------------------------------
# Hidden Populations && Neuromodulation
# --------------------------------------------------------------------------------------

hidden_pop_size = 200

left_stim_rate = 0 if TESTING else 1
right_stim_rate = left_stim_rate
stim_pop_size = hidden_pop_size
stim_weight = 5.

dopaminergic_weight = .1

# TODO: decrease the eligibility trace time constant
# Create STDP dynamics with neuromodulation
hidden_synapse_dynamics = p.STDPMechanism(
    timing_dependence=p.IzhikevichNeuromodulation(
        tau_plus=60., tau_minus=60.,
        A_plus=0.25, A_minus=0.25,
        tau_c=300., tau_d=20.),
    weight_dependence=p.MultiplicativeWeightDependence(w_min=0, w_max=3),
    weight=.5,
    neuromodulation=True)

# --------------------------------------------------------------------------------------
# Left Hidden Population
# --------------------------------------------------------------------------------------

left_hidden_pop = p.Population(hidden_pop_size, p.IF_curr_exp_izhikevich_neuromodulation,
                               label="left_hidden_pop")

# Stimulation population
left_stimulation_pop = p.Population(stim_pop_size, p.SpikeSourcePoisson(rate=left_stim_rate),
                                    label="left_stimulation_pop")
left_stim_projection = p.Projection(left_stimulation_pop, left_hidden_pop,
                                    p.OneToOneConnector(),
                                    p.StaticSynapse(weight=stim_weight))

# Create Dopaminergic connections
# p.Projection(
#     reward_pop, left_hidden_pop,
#     p.AllToAllConnector(),
#     synapse_type=p.StaticSynapse(weight=dopaminergic_weight),
#     receptor_type='reward', label='reward synapses -> left hidden')
p.Projection(
    ball_on_left_dopaminergic_pop, left_hidden_pop,
    p.AllToAllConnector(),
    synapse_type=p.StaticSynapse(weight=dopaminergic_weight),
    receptor_type='reward', label='reward ball on left synapses -> left hidden')
p.Projection(
    ball_on_right_dopaminergic_pop, left_hidden_pop,
    p.AllToAllConnector(),
    synapse_type=p.StaticSynapse(weight=dopaminergic_weight),
    receptor_type='punishment', label='punish ball on right synapses -> left hidden')

if LOAD_PREVIOUS_CONNECTIONS:
    prev_ball_x_left_conn = previous_connections[0]
    prev_paddle_left_conn = previous_connections[1]

# Create a plastic connection between Ball and Hidden neurons
ball_x_left_plastic_projection = p.Projection(
    ball_x_pop, left_hidden_pop,
    p.FromListConnector(prev_ball_x_left_conn) if LOAD_PREVIOUS_CONNECTIONS
    else
    p.AllToAllConnector(),
    synapse_type=hidden_synapse_dynamics,
    receptor_type='excitatory', label='Ball_x-Left_Hidden projection')

# Create a plastic connection between Paddle and Hidden neurons
paddle_left_plastic_projection = p.Projection(
    paddle_pop, left_hidden_pop,
    p.FromListConnector(prev_paddle_left_conn) if LOAD_PREVIOUS_CONNECTIONS
    else
    p.AllToAllConnector(),
    synapse_type=hidden_synapse_dynamics,
    receptor_type='excitatory', label='Paddle-Left_Hidden projection')

# --------------------------------------------------------------------------------------
# Right Hidden Population
# --------------------------------------------------------------------------------------

right_hidden_pop = p.Population(hidden_pop_size, p.IF_curr_exp_izhikevich_neuromodulation,
                                label="right_hidden_pop")

# Stimulation population
right_stimulation_pop = p.Population(stim_pop_size, p.SpikeSourcePoisson(rate=right_stim_rate),
                                     label="right_stimulation_pop")
right_stim_projection = p.Projection(right_stimulation_pop, right_hidden_pop,
                                     p.OneToOneConnector(),
                                     p.StaticSynapse(weight=stim_weight))

# Create Dopaminergic connections
# p.Projection(
#     reward_pop, right_hidden_pop,
#     p.AllToAllConnector(),
#     synapse_type=p.StaticSynapse(weight=dopaminergic_weight),
#     receptor_type='reward', label='reward synapses -> right hidden')
p.Projection(
    ball_on_left_dopaminergic_pop, right_hidden_pop,
    p.AllToAllConnector(),
    synapse_type=p.StaticSynapse(weight=dopaminergic_weight),
    receptor_type='punishment', label='punish ball on left synapses -> right hidden')
p.Projection(
    ball_on_right_dopaminergic_pop, right_hidden_pop,
    p.AllToAllConnector(),
    synapse_type=p.StaticSynapse(weight=dopaminergic_weight),
    receptor_type='reward', label='reward ball on right synapses -> right hidden')

if LOAD_PREVIOUS_CONNECTIONS:
    prev_ball_x_right_conn = previous_connections[2]
    prev_paddle_right_conn = previous_connections[3]

# Create a plastic connection between Ball and Hidden neurons
ball_x_right_plastic_projection = p.Projection(
    ball_x_pop, right_hidden_pop,
    p.FromListConnector(prev_ball_x_right_conn) if LOAD_PREVIOUS_CONNECTIONS
    else
    p.AllToAllConnector(),
    synapse_type=hidden_synapse_dynamics,
    receptor_type='excitatory', label='Ball_x-Right_Hidden projection')

# Create a plastic connection between Paddle and Hidden neurons
paddle_right_plastic_projection = p.Projection(
    paddle_pop, right_hidden_pop,
    p.FromListConnector(prev_paddle_right_conn) if LOAD_PREVIOUS_CONNECTIONS
    else
    p.AllToAllConnector(),
    synapse_type=hidden_synapse_dynamics,
    receptor_type='excitatory', label='Paddle-Right_Hidden projection')

# --------------------------------------------------------------------------------------
# Decision Population && Neuromodulation
# --------------------------------------------------------------------------------------

# For the decision neuron to spike it needs at least 2 input spikes
hidden_to_decision_weight = 0.085 / 2

decision_input_pop = p.Population(2, p.IF_cond_exp, label="decision_input_pop")

[left_decision_conn, right_decision_conn] = get_hidden_to_decision_connections(hidden_pop_size,
                                                                               weight=hidden_to_decision_weight)

p.Projection(left_hidden_pop, decision_input_pop, p.FromListConnector(left_decision_conn),
             p.StaticSynapse(weight=hidden_to_decision_weight))
p.Projection(right_hidden_pop, decision_input_pop, p.FromListConnector(right_decision_conn),
             p.StaticSynapse(weight=hidden_to_decision_weight))

# Connect input decision population to the game
p.Projection(decision_input_pop, breakout_pop, p.OneToOneConnector(),
             p.StaticSynapse(weight=1.))

# ----------------------------------------------------------------------------------------------------------------------
# Setup recording
# ----------------------------------------------------------------------------------------------------------------------

paddle_pop.record('spikes')
ball_x_pop.record('spikes')
left_hidden_pop.record('spikes')
right_hidden_pop.record('spikes')
decision_input_pop.record('spikes')
if RANDOM_SPIKE_INPUT:
    random_spike_input.record('spikes')
reward_pop.record('all')
ball_on_left_dopaminergic_pop.record('all')
ball_on_right_dopaminergic_pop.record('all')

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

runtime = SIMULATION_TIME
simulator = get_simulator()
print("\nLet\'s play breakout!")
p.run(runtime)

# ----------------------------------------------------------------------------------------------------------------------
# Post-Process Results and Plots
# ----------------------------------------------------------------------------------------------------------------------

print("\nSimulation Complete - Extracting Data and Post-Processing")
vis_proc.terminate()

pad_pop_spikes = paddle_pop.get_data('spikes')
ball_x_pop_spikes = ball_x_pop.get_data('spikes')
left_hidden_pop_spikes = left_hidden_pop.get_data('spikes')
right_hidden_pop_spikes = right_hidden_pop.get_data('spikes')
decision_input_pop_spikes = decision_input_pop.get_data('spikes')
if RANDOM_SPIKE_INPUT:
    random_spike_input_spikes = random_spike_input.get_data('spikes')
reward_pop_output = reward_pop.get_data()
punishment_ball_on_left_pop_output = ball_on_left_dopaminergic_pop.get_data()
punishment_ball_on_right_pop_output = ball_on_right_dopaminergic_pop.get_data()

dopaminergic_line_properties = [{'color': 'red', 'markersize': 15},
                                {'color': 'blue', 'markersize': 10},
                                {'color': 'orange', 'markersize': 5}]

Figure(
    Panel(pad_pop_spikes.segments[0].spiketrains,
          yticks=True, xticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(ball_x_pop_spikes.segments[0].spiketrains,
          yticks=True, xticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(right_hidden_pop_spikes.segments[0].spiketrains,
          yticks=True, xticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(left_hidden_pop_spikes.segments[0].spiketrains,
          yticks=True, xticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(decision_input_pop_spikes.segments[0].spiketrains,
          yticks=True, xticks=True, markersize=0.2, xlim=(0, runtime)),

    # Panel(random_spike_input_spikes.segments[0].spiketrains,
    #       yticks=True, xticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(punishment_ball_on_left_pop_output.segments[0].filter(name='gsyn_exc')[0],
          reward_pop_output.segments[0].filter(name='gsyn_exc')[0],
          punishment_ball_on_right_pop_output.segments[0].filter(name='gsyn_exc')[0],
          line_properties=dopaminergic_line_properties,
          ylabel="gsyn excitatory (mV)",
          data_labels=[ball_on_left_dopaminergic_pop.label, reward_pop.label, ball_on_right_dopaminergic_pop.label],
          yticks=True,
          xticks=True,
          xlim=(0, runtime)
          ),
    # title="Simple Breakout Example"
)

if TESTING:
    plt.show()
    print("Displayed first plot")

# Score over time plot
scores = get_scores(breakout_pop=breakout_pop, simulator=simulator)
print("Scores: {}".format(scores))

plt.figure(2)
plt.plot(scores)
plt.ylabel("score")
plt.xlabel("machine_time_step")
plt.title("Score Evolution - Neuromodulated play")

if TESTING:
    plt.show()
    print("Displayed second plot")

# ----------------------------------------------------------------------------------------------------------------------
# Save Weights and Connections
# ----------------------------------------------------------------------------------------------------------------------
print("Extracting and Saving the connections")
save_conn = []

ball_x_left_conn = clean_connection(ball_x_left_plastic_projection.get('weight', 'list'))
paddle_left_conn = clean_connection(paddle_left_plastic_projection.get('weight', 'list'))

save_conn.append(ball_x_left_conn)
save_conn.append(paddle_left_conn)

ball_x_right_conn = clean_connection(ball_x_right_plastic_projection.get('weight', 'list'))
paddle_right_conn = clean_connection(paddle_right_plastic_projection.get('weight', 'list'))

save_conn.append(ball_x_right_conn)
save_conn.append(paddle_right_conn)

if SAVE_CONNECTIONS:
    with open(FILENAME, "w") as f:
        f.write(json.dumps(save_conn))
        print("Saved the weights and connections")

# ----------------------------------------------------------------------------------------------------------------------
# End Simulation
# ----------------------------------------------------------------------------------------------------------------------
print("Completing the simulation")
p.end()
print("Simulation Completed")
