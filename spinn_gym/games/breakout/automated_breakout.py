# Copyright (c) 2019-2022 The University of Manchester
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
import spynnaker8 as p
from spinn_gym import Breakout
from .breakout_sim import (
    subsample_connection, row_col_to_input_breakout, separate_connections,
    compress_to_x_axis, generate_ball_to_hidden_pop_connections,
    generate_decision_connections)

X_RES = 160
Y_RES = 128
X_SCALE = 2
Y_SCALE = 2
X_RES_FINAL = X_RES // X_SCALE
Y_RES_FINAL = Y_RES // Y_SCALE


class AutomatedBreakout(object):

    def __init__(self):
        # Setup pyNN simulation
        p.setup(timestep=1.0)
        p.set_number_of_neurons_per_core(p.IF_cond_exp, 128)

        # Weights
        weight = 0.1

        # ----------------------------------------------------------------------------------------------------------------------
        # Breakout Population && Spike Input
        # ----------------------------------------------------------------------------------------------------------------------

        # Create breakout population and activate live output
        b1 = Breakout(x_factor=X_SCALE, y_factor=Y_SCALE, bricking=1)
        self.breakout_pop = p.Population(b1.neurons(), b1, label="breakout1")

        # Create random spike input and connect to Breakout pop to stimulate
        # paddle (and enable paddle visualisation)
        random_spike_input = p.Population(
            2, p.SpikeSourcePoisson(rate=7), label="input_connect")
        p.Projection(
            random_spike_input, self.breakout_pop, p.AllToAllConnector(),
            p.StaticSynapse(weight=1.))

        # Connect key spike injector to breakout population
        # self.key_input = p.Population(
        #     2, p.external_devices.SpikeInjector(), label="key_input")
        # p.Projection(
        #    self.key_input, self.breakout_pop, p.AllToAllConnector(),
        #    p.StaticSynapse(weight=0.1))

        # --------------------------------------------------------------------------------------
        # ON/OFF Connections
        # --------------------------------------------------------------------------------------

        [Connections_on, Connections_off] = subsample_connection(
            X_RES_FINAL, Y_RES_FINAL, 1, 1, weight, row_col_to_input_breakout)

        [Ball_on_connections, Paddle_on_connections] = \
            separate_connections(X_RES_FINAL * Y_RES_FINAL - X_RES_FINAL,
                                 Connections_on)

        [_Ball_off_connections, Paddle_off_connections] = \
            separate_connections(X_RES_FINAL * Y_RES_FINAL - X_RES_FINAL,
                                 Connections_off)

        # --------------------------------------------------------------------------------------
        # Paddle Population
        # --------------------------------------------------------------------------------------

        self.paddle_pop = p.Population(
            X_RES_FINAL, p.IF_cond_exp(), label="paddle_pop")

        p.Projection(
            self.breakout_pop, self.paddle_pop,
            p.FromListConnector(Paddle_on_connections),
            receptor_type="excitatory")
        p.Projection(
            self.breakout_pop, self.paddle_pop,
            p.FromListConnector(Paddle_off_connections),
            receptor_type="inhibitory")

        # --------------------------------------------------------------------------------------
        # Ball Position Population
        # --------------------------------------------------------------------------------------

        self.ball_pop = p.Population(
            X_RES_FINAL, p.IF_cond_exp(), label="ball_pop")

        Compressed_ball_on_connections = compress_to_x_axis(
            Ball_on_connections, X_RES_FINAL)

        p.Projection(
            self.breakout_pop, self.ball_pop,
            p.FromListConnector(Compressed_ball_on_connections),
            receptor_type="excitatory")

        # ------------------------------------------------------------------
        # Hidden Populations
        # ------------------------------------------------------------------

        self.left_hidden_pop = p.Population(
            X_RES_FINAL, p.IF_cond_exp(), label="left_hidden_pop")
        self.right_hidden_pop = p.Population(
            X_RES_FINAL, p.IF_cond_exp(), label="right_hidden_pop")

        # Project the paddle population on left/right hidden populations
        # so that it charges the neurons without spiking
        paddle_presence_weight = 0.01
        p.Projection(
            self.paddle_pop, self.left_hidden_pop, p.OneToOneConnector(),
            p.StaticSynapse(paddle_presence_weight))
        p.Projection(
            self.paddle_pop, self.right_hidden_pop, p.OneToOneConnector(),
            p.StaticSynapse(paddle_presence_weight))

        [Ball_to_left_hidden_connections, Ball_to_right_hidden_connections] = \
            generate_ball_to_hidden_pop_connections(
                pop_size=X_RES_FINAL, ball_presence_weight=0.07)

        p.Projection(
            self.ball_pop, self.left_hidden_pop,
            p.FromListConnector(Ball_to_left_hidden_connections))
        p.Projection(
            self.ball_pop, self.right_hidden_pop,
            p.FromListConnector(Ball_to_right_hidden_connections))

        # --------------------------------------------------------------------------------------
        # Decision Population
        # --------------------------------------------------------------------------------------

        self.decision_input_pop = p.Population(
            2, p.IF_cond_exp(), label="decision_input_pop")

        [Left_decision_connections, Right_decision_connections] = \
            generate_decision_connections(
                pop_size=X_RES_FINAL, decision_weight=weight)

        p.Projection(
            self.left_hidden_pop, self.decision_input_pop,
            p.FromListConnector(Left_decision_connections))
        p.Projection(
            self.right_hidden_pop, self.decision_input_pop,
            p.FromListConnector(Right_decision_connections))

        # Connect input Decision population to the game
        p.Projection(
            self.decision_input_pop, self.breakout_pop, p.OneToOneConnector(),
            p.StaticSynapse(weight=1.0))

        # --------------------------------------------------------------------------------------
        # Reward Population
        # --------------------------------------------------------------------------------------

        # Create population to receive reward signal from
        # Breakout (n0: reward, n1: punishment)
        self.receive_reward_pop = p.Population(
            2, p.IF_cond_exp(), label="receive_rew_pop")

        p.Projection(
            self.breakout_pop, self.receive_reward_pop, p.OneToOneConnector(),
            p.StaticSynapse(weight=weight))
