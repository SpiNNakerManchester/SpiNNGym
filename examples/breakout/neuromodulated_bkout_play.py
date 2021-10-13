from __future__ import print_function

import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

import spynnaker8 as p
from spinn_gym.games.breakout.breakout_sim import get_scores
from spinn_gym.games.breakout.neuromodulated_breakout import (
    NeuromodulatedBreakout)
from spinn_front_end_common.utilities.globals_variables import get_simulator


# ----------------------------------------------------------------------------------------------------------------------
# Initialise Simulation and Parameters
# ----------------------------------------------------------------------------------------------------------------------

breakout = NeuromodulatedBreakout()
breakout.paddle_pop.record("spikes")

# ----------------------------------------------------------------------------------------------------------------------
# Configure Visualiser
# ----------------------------------------------------------------------------------------------------------------------

# configure_visualiser(
#    breakout, X_RES, Y_RES, X_SCALE, Y_SCALE, start_external_visualiser)

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

Figure(
    Panel(pad_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime))
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
