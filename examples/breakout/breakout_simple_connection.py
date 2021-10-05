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

# non-SpiNNaker imports
import matplotlib.pyplot as plt
import numpy as np
from pyNN.utility.plotting import Figure, Panel
import functools
import subprocess
import sys

from examples.breakout.breakout_sim import make_simulation

# SpiNNaker imports
from spinn_front_end_common.utilities.globals_variables import get_simulator
from spinn_front_end_common.utilities.database.database_connection \
    import DatabaseConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
from spynnaker.pyNN.models.utility_models.spike_injector import \
    SpikeInjector
import spynnaker8 as p
import spinn_gym as gym

# -----------------------------------------------------------------------------
#  Globals
# -----------------------------------------------------------------------------
vis_proc = None  # Visualiser process (global)

# Game resolution and scaling
X_RESOLUTION = 160
Y_RESOLUTION = 128
X_SCALE = 2
Y_SCALE = 2
LABEL = "Breakout"

make_simulation(X_RESOLUTION, Y_RESOLUTION, X_SCALE, Y_SCALE, LABEL)

key_input_connection = p.external_devices.SpynnakerLiveSpikesConnection(
    send_labels=[key_input_label], local_port=None)
p.external_devices.add_database_socket_address(
    "localhost", key_input_connection.local_port, None)

# -----------------------------------------------------------------------------
# Configure Visualiser
# -----------------------------------------------------------------------------

d_conn = DatabaseConnection(local_port=None)

print("\nRegister visualiser process")
d_conn.add_database_callback(functools.partial(
    start_visualiser, pop_label=b1.label, xr=x_factor1, yr=y_factor1,
    xb=np.uint32(np.ceil(np.log2(X_RESOLUTION/x_factor1))),
    yb=np.uint32(np.ceil(np.log2(Y_RESOLUTION/y_factor1))),
    key_conn=key_input_connection))

p.external_devices.add_database_socket_address(
     "localhost", d_conn.local_port, None)

# -----------------------------------------------------------------------------
# Run Simulation
# -----------------------------------------------------------------------------
runtime = 1000 * 60
simulator = get_simulator()
print("\nLet\'s play breakout!")
p.run(runtime)

# -----------------------------------------------------------------------------
# Post-Process Results
# -----------------------------------------------------------------------------
print("\nSimulation Complete - Extracting Data and Post-Processing")

spike_input_spikes = spike_input.get_data('spikes')
receive_pop_spikes = receive_pop.get_data('spikes')
receive_reward_pop_output = receive_reward_pop.get_data()

figure_filename = "results.png"
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spike_input_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(receive_pop_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    Panel(receive_reward_pop_output.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[receive_reward_pop.label],
          yticks=True,
          xlim=(0, runtime)
          ),
    title="Simple Breakout Example",
    annotations="Simulated with {}".format(p.name())
)

plt.show()

scores = get_scores(breakout_pop=breakout_pop, simulator=simulator)
print("Scores: {}".format(scores))

# End simulation
p.end()
vis_proc.terminate()
print("Simulation Complete")
