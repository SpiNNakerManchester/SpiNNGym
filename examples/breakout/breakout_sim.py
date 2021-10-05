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
import sys
import subprocess
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


def separate_connections(ball_population_size, connections_on):
    # Separates the ball and pad connections in different populations
    paddle_list = []
    ball_list = []

    for idx, val in enumerate(connections_on):
        if idx < ball_population_size:
            ball_list.append(val)
        else:
            index_in_paddle_pop = idx - ball_population_size
            new_el_connection = (val[0], index_in_paddle_pop, val[2], val[3])
            paddle_list.append(new_el_connection)

    return ball_list, paddle_list


@DeprecationWarning
def map_to_one_neuron_per_paddle(
        pop_size, no_paddle_neurons, syn_weight, paddle_connections):
    # Get connections of compressed PADDLE population to one neuron each
    connections = []

    no_paddle_neurons = int(no_paddle_neurons)
    offset = no_paddle_neurons // 2

    for val in paddle_connections:
        from_neuron = val[1] - offset
        to_neuron = val[1] + offset + 1

        for neuron_no in range(from_neuron, to_neuron):
            if 0 <= neuron_no < pop_size:
                connections.append((val[0], neuron_no, syn_weight, val[3]))

    return connections


@DeprecationWarning
def create_lateral_inhibitory_paddle_connections(
        pop_size, no_paddle_neurons, syn_weight):
    lat_connections = []

    no_paddle_neurons = int(no_paddle_neurons)
    # just a precaution
    no_paddle_neurons += 2
    paddle_neurons_offset = no_paddle_neurons // 2

    # If the no_pad_neurons is even
    # then recalculate the offset
    if no_paddle_neurons % 2 == 0:
        paddle_neurons_offset -= 1

    paddle_neurons_offset *= 2

    for neuron in range(0, pop_size):
        for paddle_neuron in range(
                neuron - paddle_neurons_offset,
                neuron + paddle_neurons_offset + 1):
            if paddle_neuron != neuron and 0 <= paddle_neuron < pop_size:
                # I used to calculate the weight based on the number of
                # excitatory input connections
                new_weight = syn_weight * (
                    no_paddle_neurons - abs(neuron - paddle_neuron))
                lat_connections.append((neuron, paddle_neuron, new_weight, 1.))

    return lat_connections


def compress_to_x_axis(connections, x_resolution):
    # Get connections of compressed BALL population to the X axis
    compressed_connections = []

    for val in connections:
        new_el_connection = (val[0], val[1] % x_resolution, val[2], val[3])
        compressed_connections.append(new_el_connection)

    return compressed_connections


def generate_ball_to_hidden_pop_connections(pop_size, ball_presence_weight):
    left_connections = []
    right_connections = []

    for ball_neuron in range(0, pop_size):
        # Connect the ball neuron to all the neurons to the left of it in the
        # left hidden population
        for left_hidden_neuron in range(0, ball_neuron):
            right_connections.append(
                (ball_neuron, left_hidden_neuron, ball_presence_weight, 1.))
        # Connect the ball neuron to all the neurons to the right of it in the
        # right hidden population
        for right_hidden_neuron in range(ball_neuron + 1, pop_size):
            left_connections.append(
                (ball_neuron, right_hidden_neuron, ball_presence_weight, 1.))

    return left_connections, right_connections


def generate_decision_connections(pop_size, decision_weight):
    left_conn = []
    right_conn = []

    for neuron in range(0, pop_size):
        left_conn.append((neuron, 0, decision_weight, 1.))
        right_conn.append((neuron, 1, decision_weight, 1.))

    return left_conn, right_conn


def clean_connection(data):
    clean_conn = []
    for i in range(0, len(data.connections)):
        for c in data.connections[i]:
            new_c = (int(c[0]), int(c[1]), float(c[2]), float(c[3]))
            clean_conn.append(new_c)

    return clean_conn


def start_external_visualiser(
        database, pop_label, xr, yr, xb=8, yb=8, key_conn=None):
    _, _, _, board_address, tag = database.get_live_output_details(
        pop_label, "LiveSpikeReceiver")

    print("Calling \'start_visualiser\'")

    # Calling visualiser - must be done as process rather than on thread due to
    # OS security (Mac)
    return subprocess.Popen(
        [sys.executable,
         '../../spinn_gym/games/breakout/visualiser/visualiser.py',
         board_address,
         tag.__str__(),
         xb.__str__(),
         yb.__str__()
         ])
    # print("Visualiser proc ID: {}".format(vis_proc.pid))


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
