import spynnaker8 as p
import spinn_gym as gym

from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from spinn_front_end_common.utilities.globals_variables import get_simulator

# Replace this with whatever is being used to get anything recorded from
# the ICubVorEnv vertex itself
def get_scores(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return scores.tolist()


p.setup(timestep=1.0)

input_size = 2
rate = 20
input_pop = p.Population(input_size, p.SpikeSourcePoisson(rate=rate))

output_pop1 = p.Population(2, p.IF_cond_exp())
output_pop2 = p.Population(2, p.IF_cond_exp())

# get head_positions and head_velocities from file
head_positions = [0, 1, 2, 3]
head_velocities = [3, 2, 1, 0]

error_window_size = 10
error_value = 0
icub_vor_env_model = gym.ICubVorEnv(
    head_positions, head_velocities, error_window_size, error_value)
icub_vor_env_pop = p.Population(input_size, icub_vor_env_model)

input_pop.record('spikes')
# icub_vor_env_pop.record('spikes')
output_pop1.record('spikes')
output_pop2.record('spikes')

i2a = p.Projection(input_pop, icub_vor_env_pop, p.AllToAllConnector())

# test_rec = p.Projection(icub_vor_env_pop, icub_vor_env_pop, p.AllToAllConnector(),
#                     p.StaticSynapse(weight=0.1, delay=0.5))
i2o1 = p.Projection(icub_vor_env_pop, output_pop1, p.AllToAllConnector(),
                    p.StaticSynapse(weight=0.1, delay=0.5))
i2o2 = p.Projection(icub_vor_env_pop, output_pop2, p.OneToOneConnector(),
                    p.StaticSynapse(weight=0.1, delay=0.5))

simulator = get_simulator()

runtime = 10000
p.run(runtime)

scores = get_scores(icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)

print(scores)

spikes_in = input_pop.get_data('spikes').segments[0].spiketrains
spikes_out1 = output_pop1.get_data('spikes').segments[0].spiketrains
spikes_out2 = output_pop2.get_data('spikes').segments[0].spiketrains
Figure(
    Panel(spikes_in, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_out1, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_out2, xlabel="Time (ms)", ylabel="nID", xticks=True)
)
plt.show()

p.end()
