# Copyright (c) 2019-2022 The University of Manchester
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
from __future__ import print_function

import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

import pyNN.spiNNaker as p
from spinn_gym.games.breakout.breakout_sim import get_scores
from spinn_gym.games.breakout.neuromodulated_breakout import (
    NeuromodulatedBreakout)


# ------------------------------------------------------------------------------
# Initialise Simulation and Parameters
# ------------------------------------------------------------------------------

breakout = NeuromodulatedBreakout()
breakout.paddle_pop.record("spikes")

# ------------------------------------------------------------------------------
# Configure Visualiser
# ------------------------------------------------------------------------------

# configure_visualiser(
#    breakout, X_RES, Y_RES, X_SCALE, Y_SCALE, start_external_visualiser)

# ------------------------------------------------------------------------------
# Run Simulation
# ------------------------------------------------------------------------------

runtime = 1000 * 10
print("\nLet\'s play breakout!")
p.run(runtime)

# ------------------------------------------------------------------------------
# Post-Process Results
# ------------------------------------------------------------------------------

print("\nSimulation Complete - Extracting Data and Post-Processing")

pad_pop_spikes = breakout.paddle_pop.get_data('spikes')

Figure(
    Panel(pad_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime))
)

plt.show()

scores = get_scores(breakout_pop=breakout.breakout_pop)
print(f"Scores: {scores}")

plt.figure(2)
plt.plot(scores)
plt.ylabel("score")
plt.xlabel("machine_time_step")
plt.title("Score Evolution - Automated play")

plt.show()

# End simulation
p.end()
print("Simulation Complete")
