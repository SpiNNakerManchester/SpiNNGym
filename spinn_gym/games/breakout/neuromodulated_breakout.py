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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import pyNN.spiNNaker as p
from spinn_gym import Breakout
from .breakout_sim import (
    subsample_connection, row_col_to_input_breakout, separate_connections,
    compress_to_x_axis, compress_to_y_axis, map_to_one_neuron_per_paddle,
    create_lateral_inhibitory_paddle_connections,
    get_hidden_to_decision_connections)

X_RES = 160
Y_RES = 128
X_SCALE = 2
Y_SCALE = 2
X_RES_FINAL = X_RES // X_SCALE
Y_RES_FINAL = Y_RES // Y_SCALE


class NeuromodulatedBreakout(object):

    def __init__(self, time_scale_factor=1):
        # Setup pyNN simulation
        p.setup(timestep=1.0, time_scale_factor=time_scale_factor)
        p.set_number_of_neurons_per_core(p.IF_cond_exp, 128)

        # Weights
        weight = 0.1

        # ---------------------------------------------------------------------
        # Breakout Population && Spike Input
        # ---------------------------------------------------------------------

        # Create breakout population and activate live output
        b1 = Breakout(x_factor=X_SCALE, y_factor=Y_SCALE, bricking=1)
        self.breakout_pop = p.Population(b1.n_atoms, b1, label="breakout1")

        # self.key_input = p.Population(
        #     2, p.external_devices.SpikeInjector(), label="key_input")
        # p.external_devices.activate_live_output_to(
        #     self.key_input, self.breakout_pop)

        # --------------------------------------------------------------------
        # Reward & Punishment Population
        # --------------------------------------------------------------------

        # n0: reward, n1: punishment
        punishment_ball_on_left_conn = [(1, 0, 2, 10)]
        punishment_ball_on_right_conn = [(0, 0, 2, 10)]

        ball_on_left_dopaminergic_pop = p.Population(
            1, p.IF_cond_exp(), label="punishment_pop")
        ball_on_right_dopaminergic_pop = p.Population(
            1, p.IF_cond_exp(), label="new_dopaminergic_pop")

        p.Projection(
            self.breakout_pop, ball_on_left_dopaminergic_pop,
            p.FromListConnector(punishment_ball_on_left_conn))
        p.Projection(
            self.breakout_pop, ball_on_right_dopaminergic_pop,
            p.FromListConnector(punishment_ball_on_right_conn))

        # ---------------------------------------------------------------------
        # ON/OFF Connections
        # ---------------------------------------------------------------------

        [Connections_on, _] = subsample_connection(
            X_RES_FINAL, Y_RES_FINAL, 1, 1, weight, row_col_to_input_breakout)

        [Ball_on_connections, Paddle_on_connections] = separate_connections(
            X_RES_FINAL * Y_RES_FINAL - X_RES_FINAL, Connections_on)

        # --------------------------------------------------------------------
        # Paddle Population
        # --------------------------------------------------------------------

        # based on the size of the bat in bkout.c
        paddle_neuron_size = 30 // X_SCALE
        paddle_to_one_neuron_weight = 0.0875 / paddle_neuron_size

        Compressed_paddle_connections = map_to_one_neuron_per_paddle(
            X_RES_FINAL, paddle_neuron_size, paddle_to_one_neuron_weight,
            Paddle_on_connections)
        Lat_inh_connections = create_lateral_inhibitory_paddle_connections(
            X_RES_FINAL, paddle_neuron_size, paddle_to_one_neuron_weight / 2)

        self.paddle_pop = p.Population(
            X_RES_FINAL, p.IF_cond_exp(), label="paddle_pop")

        p.Projection(
            self.breakout_pop, self.paddle_pop,
            p.FromListConnector(Compressed_paddle_connections),
            receptor_type="excitatory")
        p.Projection(
            self.paddle_pop, self.paddle_pop,
            p.FromListConnector(Lat_inh_connections),
            receptor_type="inhibitory")

        # ---------------------------------------------------------------------
        # Ball Positions Populations
        # ---------------------------------------------------------------------

        self.ball_x_pop = p.Population(
            X_RES_FINAL, p.IF_cond_exp(), label="ball_x_pop")
        self.ball_y_pop = p.Population(
            Y_RES_FINAL, p.IF_cond_exp(), label="ball_y_pop")

        Ball_x_connections = compress_to_x_axis(
            Ball_on_connections, X_RES_FINAL)
        Ball_y_connections = compress_to_y_axis(
            Ball_on_connections, Y_RES_FINAL)

        p.Projection(
            self.breakout_pop, self.ball_x_pop,
            p.FromListConnector(Ball_x_connections),
            p.StaticSynapse(weight=weight))
        p.Projection(
            self.breakout_pop, self.ball_y_pop,
            p.FromListConnector(Ball_y_connections),
            p.StaticSynapse(weight=weight))

        # --------------------------------------------------------------------
        # Hidden Populations && Neuromodulation
        # --------------------------------------------------------------------

        hidden_pop_size = 150

        stim_rate = 3.
        stim_pop_size = hidden_pop_size
        stim_weight = 0.01

        dopaminergic_weight = .1

        # -----------------------------------------------------------------
        # Stimulation Population
        # -----------------------------------------------------------------
        stimulation_pop = p.Population(
            stim_pop_size, p.SpikeSourcePoisson(rate=stim_rate),
            label="left_stimulation_pop")

        # ----------------------------------------------------------------
        # Left Hidden Population
        # ----------------------------------------------------------------

        self.left_hidden_pop = p.Population(
            hidden_pop_size, p.IF_cond_exp(),
            label="left_hidden_pop")

        # Stimulate Left Hidden pop
        p.Projection(
            stimulation_pop, self.left_hidden_pop, p.OneToOneConnector(),
            p.StaticSynapse(weight=stim_weight))

        # Create STDP dynamics with neuromodulation
        hidden_synapse_dynamics = p.STDPMechanism(
            timing_dependence=p.SpikePairRule(
                tau_plus=30., tau_minus=30.,
                A_plus=0.02, A_minus=0.02),
            weight_dependence=p.AdditiveWeightDependence(
                w_min=0, w_max=0.5),
            weight=.5)

        # Create a plastic connection between Ball X and Hidden neurons
        p.Projection(
            self.ball_x_pop, self.left_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Ball_x-Left_Hidden projection')

        # Create a plastic connection between Ball Y and Hidden neurons
        p.Projection(
            self.ball_y_pop, self.left_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Ball_y-Left_Hidden projection')

        # Create a plastic connection between Paddle and Hidden neurons
        p.Projection(
            self.paddle_pop, self.left_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Paddle-Left_Hidden projection')

        # Create Dopaminergic connections
        p.Projection(
            ball_on_left_dopaminergic_pop, self.left_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=p.extra_models.Neuromodulation(
                weight=dopaminergic_weight, tau_c=30., tau_d=10.),
            receptor_type='reward',
            label='reward ball on left synapses -> left hidden')
        p.Projection(
            ball_on_right_dopaminergic_pop, self.left_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=p.extra_models.Neuromodulation(
                weight=dopaminergic_weight, tau_c=30., tau_d=10.),
            receptor_type='punishment',
            label='punish ball on right synapses -> left hidden')

        # --------------------------------------------------------------------
        # Right Hidden Population
        # --------------------------------------------------------------------

        self.right_hidden_pop = p.Population(
            hidden_pop_size, p.IF_cond_exp(),
            label="right_hidden_pop")

        # Stimulate Right Hidden pop
        p.Projection(
            stimulation_pop, self.right_hidden_pop,
            p.OneToOneConnector(),
            p.StaticSynapse(weight=stim_weight))

        # Create a plastic connection between Ball X and Hidden neurons
        self.ball_x_learning_proj = p.Projection(
            self.ball_x_pop, self.right_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Ball_x-Right_Hidden projection')

        # Create a plastic connection between Ball Y and Hidden neurons
        p.Projection(
            self.ball_y_pop, self.right_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Ball_y-Right_Hidden projection')

        # Create a plastic connection between Paddle and Hidden neurons
        p.Projection(
            self.paddle_pop, self.right_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Paddle-Right_Hidden projection')

        # Create Dopaminergic connections
        p.Projection(
            ball_on_left_dopaminergic_pop, self.right_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=p.extra_models.Neuromodulation(
                weight=dopaminergic_weight, tau_c=30., tau_d=10.),
            receptor_type='punishment',
            label='punish ball on left synapses -> right hidden')
        p.Projection(
            ball_on_right_dopaminergic_pop, self.right_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=p.extra_models.Neuromodulation(
                weight=dopaminergic_weight, tau_c=30., tau_d=10.),
            receptor_type='reward',
            label='reward ball on right synapses -> right hidden')

        # ------------------------------------------------------------
        # Decision Population && Neuromodulation
        # ------------------------------------------------------------

        # For the decision neuron to spike it needs at least 4 input spikes at
        # the same time
        hidden_to_decision_weight = 0.085 / 4

        self.decision_input_pop = p.Population(
            2, p.IF_cond_exp, label="decision_input_pop")

        [left_decision_conn, right_decision_conn] = \
            get_hidden_to_decision_connections(
                hidden_pop_size, weight=hidden_to_decision_weight)

        p.Projection(
            self.left_hidden_pop, self.decision_input_pop,
            p.FromListConnector(left_decision_conn),
            p.StaticSynapse(weight=hidden_to_decision_weight))
        p.Projection(
            self.right_hidden_pop, self.decision_input_pop,
            p.FromListConnector(right_decision_conn),
            p.StaticSynapse(weight=hidden_to_decision_weight))

        # Connect input decision population to the game
        p.external_devices.activate_live_output_to(
            self.decision_input_pop, self.breakout_pop)
