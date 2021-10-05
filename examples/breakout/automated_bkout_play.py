from __future__ import print_function

import functools
import matplotlib.pyplot as plt
import numpy as np
from pyNN.utility.plotting import Figure, Panel

import spynnaker8 as p
from examples.breakout.breakout_sim import (
    get_scores, start_external_visualiser)
from examples.breakout.automated_breakout import (
    AutomatedBreakout, X_RES, X_SCALE, Y_RES, Y_SCALE)
from spinn_front_end_common.utilities.globals_variables import get_simulator


# ----------------------------------------------------------------------------------------------------------------------
# Initialise Simulation and Parameters
# ----------------------------------------------------------------------------------------------------------------------

breakout = AutomatedBreakout()

# ----------------------------------------------------------------------------------------------------------------------
# Configure Visualiser
# ----------------------------------------------------------------------------------------------------------------------
key_input_connection = p.external_devices.SpynnakerLiveSpikesConnection(
    send_labels=[breakout.key_input.label], local_port=None)

print("\nRegister visualiser process")
key_input_connection.add_database_callback(functools.partial(
    start_external_visualiser, pop_label=breakout.breakout_pop.label,
    xr=X_SCALE, yr=Y_SCALE,
    xb=np.uint32(np.ceil(np.log2(X_RES / X_SCALE))),
    yb=np.uint32(np.ceil(np.log2(Y_RES / Y_SCALE))),
    key_conn=key_input_connection))

p.external_devices.add_database_socket_address(
    "localhost", key_input_connection.local_port, None)

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

pad_pop_spikes = breakout.paddle_pop.get_data('spikes')
ball_pop_spikes = breakout.ball_pop.get_data('spikes')
# left_hidden_pop_spikes = left_hidden_pop.get_data('spikes')
# right_hidden_pop_spikes = right_hidden_pop.get_data('spikes')
decision_input_pop_spikes = breakout.decision_input_pop.get_data('spikes')
# random_spike_input_spikes = random_spike_input.get_data('spikes')
receive_reward_pop_output = breakout.receive_reward_pop.get_data()

Figure(
    Panel(pad_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(ball_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(decision_input_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),

    Panel(receive_reward_pop_output.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[breakout.receive_reward_pop.label],
          yticks=True,
          xlim=(0, runtime)
          )
)

plt.show()

scores = get_scores(breakout_pop=breakout.breakout_pop, simulator=simulator)
print("Scores: {}".format(scores))

plt.figure(2)
plt.plot(scores)
plt.ylabel("score")
plt.xlabel("machine_time_step")
plt.title("Score Evolution - Automated play")

plt.show()

# End simulation
p.end()
print("Simulation Complete")
