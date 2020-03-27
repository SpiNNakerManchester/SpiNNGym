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
    idx = idx | (row << colour_bits)  # colour bit
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
            connection_list_off.append((coord_map_func(j, i, 0, row_bits),
                                        subsampidx, weight, 1.))

    return connection_list_on, connection_list_off


def paddle_and_ball_list(width, height, paddle_height=1, weight=0.1, delay=1):
    separator_index = width * (height - paddle_height)
    ball_list = []
    paddle_list = []
    for i in range(separator_index):
        ball_list.append((i, i, weight, delay))
    for i in range(separator_index, width * height):
        paddle_list.append((i, i - separator_index, weight, delay))

    return ball_list, paddle_list


def get_ball_x_projection(width, height, paddle_height=1, weight=0.1, delay=1):
    projection_list = []
    for i in range(width * (height - paddle_height)):
        projection_list.append((i, i % width, weight, delay))

    return projection_list


def get_paddle_centre_projection(width, radius=2, weight=0.1, delay=1):
    projection_list = []
    for i in range(width):
        for j in range(max(i - radius, 0), min(i + radius + 1, width)):
            projection_list.append((i, j, weight, delay))

    return projection_list


def get_paddle_lateral_connections(width, radius=2, weight=0.1, delay=1):
    projection_list = []
    for i in range(width):
        for j in range(max(i - 2 * radius, 0), min(i + 2 * radius + 1, width)):
            if i != j:
                projection_list.append((i, j, (2 * radius + 1 - abs(i - j)) * -weight * 4.5, delay))

    return projection_list
