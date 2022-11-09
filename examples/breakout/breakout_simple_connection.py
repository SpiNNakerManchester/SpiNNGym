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

# non-SpiNNaker imports
import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

# SpiNNaker imports
from spinn_gym.games.breakout.breakout_sim import get_scores
from spinn_gym.games.breakout.random_breakout import RandomBreakout
import pyNN.spiNNaker as p

breakout = RandomBreakout()

# Setup recording
breakout.spike_input.record('spikes')
breakout.receive_pop.record('spikes')
breakout.receive_reward_pop.record('gsyn_exc')

# -----------------------------------------------------------------------------
# Run Simulation
# -----------------------------------------------------------------------------
runtime = 1000 * 60
print("\nLet\'s play breakout!")
p.run(runtime)

# -----------------------------------------------------------------------------
# Post-Process Results
# -----------------------------------------------------------------------------
print("\nSimulation Complete - Extracting Data and Post-Processing")

spike_input_spikes = breakout.spike_input.get_data('spikes')
receive_pop_spikes = breakout.receive_pop.get_data('spikes')
receive_reward_pop_output = breakout.receive_reward_pop.get_data('gsyn_exc')

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spike_input_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(receive_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(receive_reward_pop_output.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[breakout.receive_reward_pop.label],
          yticks=True,
          xlim=(0, runtime)
          ),
    title="Simple Breakout Example",
    annotations="Simulated with {}".format(p.name())
)

plt.show()

scores = get_scores(breakout_pop=breakout.breakout_pop)
print("Scores: {}".format(scores))

# End simulation
p.end()
print("Simulation Complete")
