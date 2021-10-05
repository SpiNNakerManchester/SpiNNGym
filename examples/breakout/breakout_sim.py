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
import numpy
from spinn_gym.games.breakout.breakout import Breakout
import spynnaker8 as p


def get_scores(breakout_pop, simulator):
    b_vertex = breakout_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager)

    return scores.tolist()


def row_col_to_input_breakout(row, col, is_on_input, row_bits, event_bits=1,
                              colour_bits=2, row_start=0):
    row_bits = numpy.uint32(row_bits)
    idx = numpy.uint32(0)

    if is_on_input:
        idx = 1

    row += row_start
    idx = idx | (row << (colour_bits))  # colour bit
    idx = idx | (col << (row_bits + colour_bits))

    # add two to allow for special event bits
    idx = idx + 2

    return idx


def subsample_connection(x_res, y_res, subsamp_factor_x, subsamp_factor_y,
                         weight, coord_map_func):
    # subY_BITS=int(np.ceil(np.log2(y_res/subsamp_factor)))
    connection_list_on = []
    connection_list_off = []

    sx_res = int(x_res) // int(subsamp_factor_x)
    row_bits = int(numpy.ceil(numpy.log2(y_res)))
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i // subsamp_factor_x
            sj = j // subsamp_factor_y

            # ON channels
            subsampidx = sj * sx_res + si
            connection_list_on.append((coord_map_func(j, i, 1, row_bits),
                                       subsampidx, weight, 1.))

            # OFF channels only on segment borders
            connection_list_off.append((coord_map_func(j, i, 0, row_bits),
                                        subsampidx, weight, 1.))

    return connection_list_on, connection_list_off


def make_simulation(
        x_res=160, y_res=128, x_scale=2, y_scale=2, breakout_label="Breakout",
        key_input_label="key_input", send_key_poisson=True):

    # Setup pyNN simulation
    p.setup(timestep=1.0)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 128)

    # -----------------------------------------------------------------------------
    # Create Spiking Neural Network
    # -----------------------------------------------------------------------------

    # Create breakout population and activate live output
    b1 = Breakout(x_factor=x_scale, y_factor=y_scale, bricking=1)
    breakout_pop = p.Population(b1.neurons(), b1, label=breakout_label)

    # Live output the breakout population
    p.external_devices.activate_live_output_for(breakout_pop)

    # Connect key spike injector to breakout population
    key_input = p.Population(
        2, p.external_devices.SpikeInjector, label=key_input_label)
    p.Projection(
        key_input, breakout_pop, p.AllToAllConnector(),
        p.StaticSynapse(weight=0.1))

    # Create random spike input and connect to Breakout pop to stimulate paddle
    # (and enable paddle visualisation)
    spike_input = p.Population(2, p.SpikeSourcePoisson(rate=2),
                               label="input_connect")
    p.Projection(
        spike_input, breakout_pop, p.AllToAllConnector(),
        p.StaticSynapse(weight=0.1))

    weight = 0.1
    [Connections_on, _] = subsample_connection(
        x_res / x_scale, y_res / y_scale, 1, 1, weight,
        row_col_to_input_breakout)

    # Create population of neurons to receive input from Breakout
    receive_pop_size = int(x_res / x_scale) * int(y_res / y_scale)
    receive_pop = p.Population(receive_pop_size, p.IF_cond_exp(),
                               label="receive_pop")
    p.Projection(
        breakout_pop, receive_pop, p.FromListConnector(Connections_on),
        p.StaticSynapse(weight=weight))

    # Create population to receive reward signal from Breakout
    # (n0: rew, n1: pun)
    receive_reward_pop = p.Population(
        2, p.IF_cond_exp(), label="receive_rew_pop")
    p.Projection(
        breakout_pop, receive_reward_pop, p.OneToOneConnector(),
        p.StaticSynapse(weight=0.1 * weight))

    # Setup recording
    spike_input.record('spikes')
    receive_pop.record('spikes')
    receive_reward_pop.record('all')
