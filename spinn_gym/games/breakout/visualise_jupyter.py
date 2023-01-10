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
import functools
import threading
import matplotlib.pyplot as plt
import numpy as np
import time

import pyNN.spiNNaker as p
from spinn_gym.games.breakout.visualiser.visualiser import Visualiser

try:
    from IPython import display
except ModuleNotFoundError:
    print("WARNING: Not in an IPython Environment;"
          " this visualisation code won't work!")


def start_vis_thread(database, pop_label, vis):
    _, _, _, board_address, tag = database.get_live_output_details(
        pop_label, "LiveSpikeReceiver")
    vis.set_remote_end(board_address, tag)


def start_visualiser(vis, display_handle):
    refresh_time = 0.01
    while vis.running:
        if vis.update():
            display_handle.update(plt.gcf())
        time.sleep(refresh_time)


def stop_visualiser(label, conn, vis, display_handle):
    # pylint: disable=unused-argument
    vis.close()
    vis.update()
    display_handle.update(plt.gcf())
    print("Visualiser closed")


def handle_vis_spikes(label, time, neuron_ids, vis):
    # pylint: disable=unused-argument, redefined-outer-name
    vis.handle_breakout_spikes(time, neuron_ids)


def handle_live_spikes(_label, _time, _neuron_ids, _vis):
    _vis.handle_live_spikes(_label, _time, _neuron_ids)


def jupyter_visualiser(
        breakout, x_res, x_scale, y_res, y_scale, live_spikes_pops=None):
    live_pop_labels = []
    if live_spikes_pops:
        live_pop_labels = [pop.label for pop in live_spikes_pops]

    vis_connection = p.external_devices.SpynnakerLiveSpikesConnection(
        local_port=None, receive_labels=[
            breakout.breakout_pop.label, *live_pop_labels])

    p.external_devices.activate_live_output_for(
        breakout.breakout_pop,
        database_notify_port_num=vis_connection.local_port)
    if live_spikes_pops:
        for pop in live_spikes_pops:
            p.external_devices.activate_live_output_for(
                pop, database_notify_port_num=vis_connection.local_port)
    else:
        live_spikes_pops = []

    # Create visualiser
    xb = np.uint32(np.ceil(np.log2(x_res / x_scale)))
    yb = np.uint32(np.ceil(np.log2(y_res / y_scale)))
    vis = Visualiser(
        x_factor=2, y_factor=2, x_bits=xb, y_bits=yb,
        live_pops=live_spikes_pops)
    display.clear_output(wait=True)
    vis.update()
    display_handle = display.display(plt.gcf(), display_id=True)

    vis_connection.add_receive_callback(
        breakout.breakout_pop.label,
        functools.partial(handle_vis_spikes, vis=vis))
    vis_connection.add_pause_stop_callback(
        breakout.breakout_pop.label,
        functools.partial(stop_visualiser, vis=vis,
                          display_handle=display_handle))
    for label in live_pop_labels:
        vis_connection.add_receive_callback(
            label, functools.partial(handle_live_spikes, vis=vis))

    vis_thread = threading.Thread(target=start_visualiser,
                                  args=[vis, display_handle])
    vis_thread.start()
