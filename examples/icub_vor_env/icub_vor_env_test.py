import spynnaker8 as p
import spinn_gym as gym

from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
import os
from spinn_front_end_common.utilities.globals_variables import get_simulator

# Examples of get functions for variables
def get_error(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    error = b_vertex.get_data(
        'error', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return error.tolist()


def get_l_count(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    left_count = b_vertex.get_data(
        'L_count', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return left_count.tolist()


def get_r_count(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    right_count = b_vertex.get_data(
        'R_count', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return right_count.tolist()


def get_head_pos(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    head_positions = b_vertex.get_data(
        'head_pos', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return head_positions.tolist()


def get_head_vel(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    head_velocities = b_vertex.get_data(
        'head_vel', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return head_velocities.tolist()


# Setup
p.setup(timestep=1.0)

# Build input SSP and output population
input_size = 2
rate = 20
input_pop = p.Population(input_size, p.SpikeSourcePoisson(rate=rate))

output_pop = p.Population(2, p.IF_cond_exp())

# get head_positions and head_velocities from file (1000 samples)
base_dir = "./"
head_pos = np.loadtxt(os.path.join(
    base_dir, "normalised_head_positions_1000.csv"))
head_vel = np.loadtxt(os.path.join(
    base_dir, "normalised_head_velocities_1000.csv"))

# perfect eye positions and velocities are exactly out of phase with head
perfect_eye_pos = np.concatenate((head_pos[500:], head_pos[:500]))
perfect_eye_vel = np.concatenate((head_vel[500:], head_vel[:500]))

# build ICubVorEnv model pop
error_window_size = 10
icub_vor_env_model = gym.ICubVorEnv(
    head_pos, head_vel, perfect_eye_vel, perfect_eye_pos, error_window_size)
icub_vor_env_pop = p.Population(input_size, icub_vor_env_model)

# Set recording for input and output pop (env pop records by default)
input_pop.record('spikes')
output_pop.record('spikes')

# Input -> ICubVorEnv projection
i2a = p.Projection(input_pop, icub_vor_env_pop, p.AllToAllConnector())

# ICubVorEnv -> output projection
i2o2 = p.Projection(icub_vor_env_pop, output_pop, p.OneToOneConnector(),
                    p.StaticSynapse(weight=0.1, delay=1.0))

# Store simulator and run
simulator = get_simulator()
runtime = 10000
p.run(runtime)

# Get the data from the ICubVorEnv pop
errors = get_error(icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)
l_counts = get_l_count(icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)
r_counts = get_r_count(icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)
rec_head_pos = get_head_pos(
    icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)
rec_head_vel = get_head_vel(
    icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)

# get the spike data from input and output and plot
spikes_in = input_pop.get_data('spikes').segments[0].spiketrains
spikes_out = output_pop.get_data('spikes').segments[0].spiketrains
Figure(
    Panel(spikes_in, xlabel="Time (ms)", ylabel="nID", xticks=True),
    Panel(spikes_out, xlabel="Time (ms)", ylabel="nID", xticks=True)
)
plt.show()

# plot the data from the ICubVorEnv pop
x_plot = [(n) for n in range(0, runtime, error_window_size)]
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.plot(x_plot, rec_head_pos, label="head position")
plt.plot(x_plot, rec_head_vel, label="head velocity")
# plt.plot(perfect_eye_pos, label="eye position", ls='--')
# plt.plot(perfect_eye_vel, label="eye velocity", ls='--')
plt.legend(loc="best")
plt.xlim([0, runtime])

plt.subplot(3, 1, 2)
plt.plot(x_plot, errors, label="errors")
plt.legend(loc="best")
plt.xlim([0, runtime])

plt.subplot(3, 1, 3)
plt.plot(x_plot, l_counts, 'bo', label="l_counts")
plt.plot(x_plot, r_counts, 'ro', label="r_counts")
plt.legend(loc="best")
plt.xlim([0, runtime])
plt.show()

p.end()
