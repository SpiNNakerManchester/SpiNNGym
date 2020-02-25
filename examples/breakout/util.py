import numpy as np


def get_scores(breakout_pop, simulator):
    b_vertex = breakout_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager,
        simulator.machine_time_step)

    return scores.tolist()


def row_col_to_input_breakout(row, col, is_on_input, row_bits, event_bits=1,
                              colour_bits=2, row_start=0):
    row_bits = np.uint32(row_bits)
    idx = np.uint32(0)

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
    row_bits = int(np.ceil(np.log2(y_res)))
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i // subsamp_factor_x
            sj = j // subsamp_factor_y

            # ON channels
            subsampidx = sj * sx_res + si
            connection_list_on.append((coord_map_func(j, i, 1, row_bits),
                                       subsampidx, weight, 1.))

            # OFF channels only on segment borders
            # if((j+1)%(y_res/subsamp_factor)==0 or
            # (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
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


def map_to_one_neuron_per_paddle(pop_size, no_paddle_neurons, syn_weight, paddle_connections):
    # Get connections of compressed PADDLE population to one neuron each
    connections = []

    no_paddle_neurons = int(no_paddle_neurons)
    offset = no_paddle_neurons // 2

    for idx, val in enumerate(paddle_connections):
        from_neuron = val[1] - offset
        to_neuron = val[1] + offset + 1

        for neuron_no in range(from_neuron, to_neuron):
            if 0 <= neuron_no < pop_size:
                connections.append((val[0], neuron_no, syn_weight, val[3]))

    return connections


def create_lateral_inhibitory_paddle_connections(pop_size, no_paddle_neurons, syn_weight):
    lat_connections = []

    paddle_neurons_offset = no_paddle_neurons // 2

    # If the no_pad_neurons is even
    # then recalculate the offset
    if no_paddle_neurons % 2 == 0:
        paddle_neurons_offset -= 1

    paddle_neurons_offset *= 2

    for neuron in range(0, pop_size):
        for paddle_neuron in range(neuron - paddle_neurons_offset, neuron + paddle_neurons_offset + 1):
            if paddle_neuron != neuron and 0 <= paddle_neuron < pop_size:
                # I used to calculate the weight based on the number of excitatory input connections
                new_weight = syn_weight * 4 * (paddle_neurons_offset + 1 - abs(neuron - paddle_neuron))
                lat_connections.append((neuron, paddle_neuron, new_weight, 1.))

    return lat_connections


def compress_to_x_axis(connections, x_resolution):
    # Get connections of compressed BALL population to the X axis
    compressed_connections = []

    for idx, val in enumerate(connections):
        new_el_connection = (val[0], val[1] % x_resolution, val[2], val[3])
        compressed_connections.append(new_el_connection)

    return compressed_connections


def compress_to_y_axis(connections, y_resolution):
    # Get connections of compressed BALL population to the Y axis
    compressed_connections = []

    for idx, val in enumerate(connections):
        new_el_connection = (val[0], val[1] // y_resolution, val[2], val[3])
        compressed_connections.append(new_el_connection)

    return compressed_connections


def generate_ball_to_hidden_pop_connections(pop_size, ball_presence_weight):
    left_connections = []
    right_connections = []

    for ball_neuron in range(0, pop_size):
        # Connect the ball neuron to all the neurons to the left of it in the left hidden population
        for left_hidden_neuron in range(0, ball_neuron):
            right_connections.append((ball_neuron, left_hidden_neuron, ball_presence_weight, 1.))
        # Connect the ball neuron to all the neurons to the right of it in the right hidden population
        for right_hidden_neuron in range(ball_neuron + 1, pop_size):
            left_connections.append((ball_neuron, right_hidden_neuron, ball_presence_weight, 1.))

    return left_connections, right_connections


def generate_decision_connections(pop_size, decision_weight):
    left_conn = []
    right_conn = []

    for neuron in range(0, pop_size):
        left_conn.append((neuron, 0, decision_weight, 1.))
        right_conn.append((neuron, 1, decision_weight, 1.))

    return left_conn, right_conn


def get_hidden_to_decision_connections(pop_size, weight):
    # Connect all elements from one pop to 0 for left and 1 for right

    return [(idx, 0, weight, 1.0) for idx in range(0, pop_size)], \
           [(idx, 1, weight, 1.0) for idx in range(0, pop_size)]


def clean_connection(data):
    clean_conn = []
    for i in range(0, len(data.connections)):
        for c in data.connections[i]:
            new_c = (int(c[0]), int(c[1]), float(c[2]), float(c[3]))
            clean_conn.append(new_c)

    return clean_conn
