# Copyright (c) 2019-2021 The University of Manchester
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

import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

import pyNN.spiNNaker as p
from spinn_gym.games.breakout.breakout_sim import get_scores
from spinn_gym.games.breakout.automated_breakout import AutomatedBreakout
from spinn_front_end_common.utilities.globals_variables import get_simulator


# ----------------------------------------------------------------------------------------------------------------------
# Initialise Simulation and Parameters
# ----------------------------------------------------------------------------------------------------------------------

breakout = AutomatedBreakout()
breakout.paddle_pop.record("spikes")
breakout.ball_pop.record("spikes")
breakout.decision_input_pop.record("spikes")
breakout.left_hidden_pop.record("spikes")
breakout.right_hidden_pop.record("spikes")
breakout.receive_reward_pop.record("gsyn_exc")

# ----------------------------------------------------------------------------------------------------------------------
# Configure Visualiser
# ----------------------------------------------------------------------------------------------------------------------
#
# configure_visualiser(
#     breakout, X_RES, Y_RES, X_SCALE, Y_SCALE, start_external_visualiser)

# ----------------------------------------------------------------------------------------------------------------------
# Run Simulation
# ----------------------------------------------------------------------------------------------------------------------

runtime = 1000 * 10
simulator = get_simulator()
print("\nLet\'s play breakout!")
p.run(runtime)

# ----------------------------------------------------------------------------------------------------------------------
# Post-Process Results
# ----------------------------------------------------------------------------------------------------------------------

print("\nSimulation Complete - Extracting Data and Post-Processing")

pad_pop_spikes = breakout.paddle_pop.get_data('spikes')
ball_pop_spikes = breakout.ball_pop.get_data('spikes')
decision_input_pop_spikes = breakout.decision_input_pop.get_data('spikes')
left_hidden_pop_spikes = breakout.left_hidden_pop.get_data("spikes")
right_hidden_pop_spikes = breakout.right_hidden_pop.get_data("spikes")
receive_reward_pop_output = breakout.receive_reward_pop.get_data('gsyn_exc')

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
