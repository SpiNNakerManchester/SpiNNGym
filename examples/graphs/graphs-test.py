from random import random
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Helper Functions
# -----------------------------------------------------------------------------


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


def get_centre_to_hidden_projection(offset, size):
    projection_list = []
    gap = 2 if size % 2 == 0 else 1
    for i in range(size):
        projection_list.append((offset + i, offset + i, random() - 0.5))
        projection_list.append((offset + i, offset + size + gap + i, random() - 0.5))

    return projection_list


# -----------------------------------------------------------------------------
#  Create Network
# -----------------------------------------------------------------------------

pop_size = 10

paddle_pop = range(0, pop_size)
paddle_centre_pop = range(pop_size, 2 * pop_size)
nodes = list(paddle_pop) + list(paddle_centre_pop)

G = nx.Graph()
G.add_nodes_from(paddle_pop)
G.add_nodes_from(paddle_centre_pop)

pos = {}
for i in range(0, pop_size):
    pos[i] = [i + 1, 0]

for i in range(pop_size, 2 * pop_size):
    pos[i] = [i - (pop_size - 1), 1]

for i in range(2 * pop_size, 3 * pop_size):
    pos[i] = [i - (pop_size - 1), 1]

connections = get_centre_to_hidden_projection(pop_size, pop_size)
print(connections)

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
