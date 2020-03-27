import json
from random import random
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Helper Functions
# -----------------------------------------------------------------------------


def get_population_size(connections):
    min_index = 0
    for connection in connections:
        if connection[0] > min_index:
            min_index = connection[0]

    return min_index + 1


def get_diff_connections(connections_before, connections_after):
    diff_connections = []
    for index, connection in enumerate(connections_before):
        weight_diff = connections_after[index][2] - connection[2]
        diff_connections.append((connection[0], connection[1], weight_diff))

    return diff_connections


def potentiation_and_depression_list(connections):
    potentiation_list = []
    unaffected_list = []
    depression_list = []

    for connection in connections:
        if connection[2] < 0:
            depression_list.append(connection)
        elif connection[2] == 0:
            unaffected_list.append(connection)
        else:
            potentiation_list.append(connection)

    return potentiation_list, unaffected_list, depression_list


# For testing (without a file), this method generates random connections, for demo.
def get_random_connections(width, radius=2):
    projection_list = []
    for i in range(width):
        for j in range(max(i - radius, 0), min(i + radius + 1, width)):
            projection_list.append((i, width + j, random()))

    return projection_list


# -----------------------------------------------------------------------------
#  Create Network
# -----------------------------------------------------------------------------

BEFORE_FILENAME = "before.json"
AFTER_FILENAME = "after.json"

with open(BEFORE_FILENAME, "r") as f:
    file_before = json.loads(f.read())
    weights_before = file_before['weights']

with open(AFTER_FILENAME, "r") as f:
    file_after = json.loads(f.read())
    weights_after = file_after['weights']

pop_size = get_population_size(weights_before)

from_population = range(0, pop_size)
to_population = range(pop_size, 2 * pop_size)
nodes = list(from_population) + list(to_population)

G = nx.Graph()
G.add_nodes_from(from_population)
G.add_nodes_from(to_population)

pos = {}
for i in range(0, pop_size):
    pos[i] = [i + 1, 0]

for i in range(pop_size, 2 * pop_size):
    pos[i] = [i - (pop_size - 1), 1]

connections = get_diff_connections(weights_before, weights_after)
pl, ul, dl = potentiation_and_depression_list(connections)


# -----------------------------------------------------------------------------
#  Draw Network
# -----------------------------------------------------------------------------

nx.draw_networkx(G, pos, nodelist=nodes, node_size=100, node_color='#c4daef', with_labels=False)
nx.draw_networkx_edges(G, pos, pl, edge_color='green')
nx.draw_networkx_edges(G, pos, ul, edge_color='black')
nx.draw_networkx_edges(G, pos, dl, edge_color='red')

plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
plt.ylim(-0.25, 1.25)
plt.axis('off')
