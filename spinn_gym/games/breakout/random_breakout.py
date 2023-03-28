# Copyright (c) 2019 The University of Manchester
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
import pyNN.spiNNaker as p
import spynnaker8 as p
from spinn_gym import Breakout
from .breakout_sim import (
    subsample_connection, row_col_to_input_breakout)

X_RES = 160
Y_RES = 128
X_SCALE = 2
Y_SCALE = 2
X_RES_FINAL = X_RES // X_SCALE
Y_RES_FINAL = Y_RES // Y_SCALE


class RandomBreakout(object):

    def __init__(self):
        # Setup pyNN simulation
        p.setup(timestep=1.0)
        p.set_number_of_neurons_per_core(p.IF_cond_exp, 128)

        # ------------------------------------------------------------
        # Create Spiking Neural Network
        # ------------------------------------------------------------

        # Create breakout population and activate live output
        b1 = Breakout(x_factor=X_SCALE, y_factor=Y_SCALE, bricking=1)
        self.breakout_pop = p.Population(b1.n_atoms, b1, label="breakout1")

        # Connect key spike injector to breakout population
        # self.key_input = p.Population(
        #     2, p.external_devices.SpikeInjector(), label="key_input")
        # p.external_devices.activate_live_output_to(
        #     self.key_input, self.breakout_pop)

        # Create random spike input and connect to Breakout pop to stimulate
        # paddle (and enable paddle visualisation)
        self.spike_input = p.Population(
            2, p.SpikeSourcePoisson(rate=2), label="input_connect")
        p.external_devices.activate_live_output_to(
            self.spike_input, self.breakout_pop)
        self.breakout_pop._vertex.source_vertex = \
            self.spike_input._vertex

        weight = 0.1
        [Connections_on, _] = subsample_connection(
            X_RES / X_SCALE, Y_RES / Y_SCALE, 1, 1, weight,
            row_col_to_input_breakout)

        # Create population of neurons to receive input from Breakout
        receive_pop_size = int(X_RES / X_SCALE) * int(Y_RES / Y_SCALE)
        self.receive_pop = p.Population(
            receive_pop_size, p.IF_cond_exp(), label="receive_pop")
        p.Projection(
            self.breakout_pop, self.receive_pop,
            p.FromListConnector(Connections_on),
            p.StaticSynapse(weight=weight))

        # Create population to receive reward signal from Breakout
        # (n0: rew, n1: pun)
        self.receive_reward_pop = p.Population(
            2, p.IF_cond_exp(), label="receive_rew_pop")
        p.Projection(
            self.breakout_pop, self.receive_reward_pop, p.OneToOneConnector(),
            p.StaticSynapse(weight=0.1 * weight))
