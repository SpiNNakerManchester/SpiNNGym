import json
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "connections.json"


def prepare_conn_for_plotting(connections, dimensions):
    weights_sum = np.zeros(dimensions)
    weights_count = np.zeros(dimensions) + 1

    for c in connections:
        weights_sum[c[0], c[1]] += c[2]
        weights_count[c[0], c[1]] += 1

    return np.divide(weights_sum, weights_count)


with open(FILENAME, "r") as f:
    previous_connections = json.loads(f.read())

ball_x_left_conn = previous_connections[0]
ball_y_left_conn = previous_connections[1]
paddle_left_conn = previous_connections[2]

ball_x_right_conn = previous_connections[3]
ball_y_right_conn = previous_connections[4]
paddle_right_conn = previous_connections[5]

left_decision_conn = previous_connections[6]
right_decision_conn = previous_connections[7]

fig, axs = plt.subplots(3)
fig.suptitle("Weights to Left Hidden")
axs[0].set_title("Ball X")
axs[0].matshow(prepare_conn_for_plotting(ball_x_left_conn, (80, 500)))
axs[1].set_title("Ball Y")
axs[1].matshow(prepare_conn_for_plotting(ball_y_left_conn, (64, 500)))
axs[2].set_title("Paddle X")
axs[2].matshow(prepare_conn_for_plotting(paddle_left_conn, (80, 500)))

plt.show()

fig, axs = plt.subplots(3)
fig.suptitle("Weights to Right Hidden")
axs[0].set_title("Ball X")
axs[0].matshow(prepare_conn_for_plotting(ball_x_right_conn, (80, 500)))
axs[1].set_title("Ball Y")
axs[1].matshow(prepare_conn_for_plotting(ball_y_right_conn, (64, 500)))
axs[2].set_title("Paddle X")
axs[2].matshow(prepare_conn_for_plotting(paddle_right_conn, (80, 500)))

plt.show()

fig, axs = plt.subplots(2)
fig.suptitle("Weights to Decision Hidden")
axs[0].set_title("Left Hidden")
axs[0].plot(prepare_conn_for_plotting(left_decision_conn, (500, 2)), 'o')
axs[1].set_title("Right Hidden")
axs[1].plot(prepare_conn_for_plotting(right_decision_conn, (500, 2)), 'o')

plt.show()

