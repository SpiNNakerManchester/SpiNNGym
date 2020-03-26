import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

FILENAME = "connections.json"
hidden_pop_size = 150
X_RES = 80
Y_RES = 64


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
paddle_left_conn = previous_connections[1]

ball_x_right_conn = previous_connections[2]
paddle_right_conn = previous_connections[3]

fig, axs = plt.subplots(1, 2)
fig.suptitle("Weights to Left Hidden")

axs[0].set_title("Ball X")
axs0 = axs[0].matshow(ndimage.rotate(prepare_conn_for_plotting(ball_x_left_conn, (X_RES, hidden_pop_size)), 90))
fig.colorbar(axs0, ax=axs[0])

axs[1].set_title("Paddle X")
axs1 = axs[1].matshow(ndimage.rotate(prepare_conn_for_plotting(paddle_left_conn, (X_RES, hidden_pop_size)), 90))
fig.colorbar(axs1, ax=axs[1])

plt.show()

fig, axs = plt.subplots(1, 2)
fig.suptitle("Weights to Right Hidden")

axs[0].set_title("Ball X")
axs0 = axs[0].matshow(prepare_conn_for_plotting(ball_x_right_conn, (X_RES, hidden_pop_size)))
fig.colorbar(axs0, ax=axs[0])

axs[1].set_title("Paddle X")
axs1 = axs[1].matshow(prepare_conn_for_plotting(paddle_right_conn, (X_RES, hidden_pop_size)))
fig.colorbar(axs1, ax=axs[1])

plt.show()
