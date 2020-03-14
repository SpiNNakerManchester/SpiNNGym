import json
import matplotlib.pyplot as plt
import numpy as np

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

fig, axs = plt.subplots(2)
fig.suptitle("Weights to Left Hidden")
axs[0].set_title("Ball X")
axs0 = axs[0].matshow(prepare_conn_for_plotting(ball_x_left_conn, (X_RES, hidden_pop_size)))
fig.colorbar(axs0, ax=axs[0])
axs[1].set_title("Paddle X")
axs1 = axs[1].matshow(prepare_conn_for_plotting(paddle_left_conn, (X_RES, hidden_pop_size)))
fig.colorbar(axs1, ax=axs[1])

plt.show()

fig, axs = plt.subplots(2)
fig.suptitle("Weights to Right Hidden")
axs[0].set_title("Ball X")
axs0 = axs[0].matshow(prepare_conn_for_plotting(ball_x_right_conn, (X_RES, hidden_pop_size)))
fig.colorbar(axs0, ax=axs[0])
axs[1].set_title("Paddle X")
axs1 = axs[1].matshow(prepare_conn_for_plotting(paddle_right_conn, (X_RES, hidden_pop_size)))
fig.colorbar(axs1, ax=axs[1])

plt.show()

# fig, axs = plt.subplots(1)
# fig.suptitle("Scores over plays")
#
# axs.plot([72, 72, 57, 52, 61, 45, 75, 52, 88, 75, 57, 46,
#           61, 53, 62, 68, 51, 54, 67, 52, 65, 45, 32, 64,
#           52, 74, 36, 54, 41, 76, 43, 72, 31, 73, 71, 65,
#           50, 78, 47, 45, 45, 41, 25])
# # axs[0].xlabel('5 min plays')
# # axs[0].ylabel('score')
#
# plt.show()
