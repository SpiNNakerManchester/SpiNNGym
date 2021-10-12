import json
import spynnaker8 as p
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
    def __init__(self, previous_connections=None):
        p.setup(timestep=1.0)
        p.set_number_of_neurons_per_core(p.IF_cond_exp, 32)
        p.set_number_of_neurons_per_core(p.IF_curr_exp, 16)

        # Weights
        weight = 0.1

        # --------------------------------------------------------------------
        # Load previous data
        # --------------------------------------------------------------------
        load_previous_connections = previous_connections is not None
        if load_previous_connections:
            with open(previous_connections, "r") as f:
                previous_connections = json.loads(f.read())

        # --------------------------------------------------------------------
        # Breakout Population && Spike Input
        # --------------------------------------------------------------------
        b1 = Breakout(x_factor=X_SCALE, y_factor=Y_SCALE, bricking=1)
        self.breakout_pop = p.Population(b1.neurons(), b1, label="breakout1")

        # ex is the external device plugin manager
        p.external_devices.activate_live_output_for(self.breakout_pop)

        self.key_input = p.Population(
            2, p.external_devices.SpikeInjector(), label="key_input")
        p.Projection(
            self.key_input, self.breakout_pop, p.AllToAllConnector(),
            p.StaticSynapse(weight=0.1))

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

        # --------------------------------------------------------------------
        # ON/OFF Connections
        # --------------------------------------------------------------------

        [Connections_on, _] = subsample_connection(
            X_RES, Y_RES, 1, 1, weight, row_col_to_input_breakout)

        [Ball_on_connections, Paddle_on_connections] = separate_connections(
            X_RES * Y_RES - X_RES, Connections_on)

        # --------------------------------------------------------------------
        # Paddle Population
        # --------------------------------------------------------------------

        # based on the size of the bat in bkout.c
        paddle_neuron_size = 30 // X_SCALE
        paddle_to_one_neuron_weight = 0.0875 / paddle_neuron_size

        Compressed_paddle_connections = map_to_one_neuron_per_paddle(
            X_RES, paddle_neuron_size, paddle_to_one_neuron_weight,
            Paddle_on_connections)
        Lat_inh_connections = create_lateral_inhibitory_paddle_connections(
            X_RES, paddle_neuron_size, paddle_to_one_neuron_weight / 2)

        self.paddle_pop = p.Population(
            X_RES, p.IF_cond_exp(), label="paddle_pop")
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

        ball_x_pop = p.Population(X_RES, p.IF_cond_exp(), label="ball_x_pop")
        ball_y_pop = p.Population(Y_RES, p.IF_cond_exp(), label="ball_y_pop")

        Ball_x_connections = compress_to_x_axis(Ball_on_connections, X_RES)
        Ball_y_connections = compress_to_y_axis(Ball_on_connections, Y_RES)

        p.Projection(
            self.breakout_pop, ball_x_pop,
            p.FromListConnector(Ball_x_connections),
            p.StaticSynapse(weight=weight))
        p.Projection(
            self.breakout_pop, ball_y_pop,
            p.FromListConnector(Ball_y_connections),
            p.StaticSynapse(weight=weight))

        # --------------------------------------------------------------------
        # Hidden Populations && Neuromodulation
        # --------------------------------------------------------------------

        hidden_pop_size = 150

        stim_rate = 3.
        stim_pop_size = hidden_pop_size
        stim_weight = 5.

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

        left_hidden_pop = p.Population(
            hidden_pop_size, p.IF_curr_exp(),
            label="left_hidden_pop")

        # Stimulate Left Hidden pop
        p.Projection(
            stimulation_pop, left_hidden_pop, p.OneToOneConnector(),
            p.StaticSynapse(weight=stim_weight))

        if load_previous_connections:
            prev_ball_x_left_conn = previous_connections[0]
            prev_ball_y_left_conn = previous_connections[1]
            prev_paddle_left_conn = previous_connections[2]

        # Create STDP dynamics with neuromodulation
        hidden_synapse_dynamics = p.STDPMechanism(
            timing_dependence=p.SpikePairRule(
                tau_plus=30., tau_minus=30.,
                A_plus=0.25, A_minus=0.25),
            weight_dependence=p.AdditiveWeightDependence(
                w_min=0, w_max=2.0),
            weight=.5)

        # Create a plastic connection between Ball X and Hidden neurons
        p.Projection(
            ball_x_pop, left_hidden_pop,
            p.FromListConnector(prev_ball_x_left_conn)
            if load_previous_connections else p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Ball_x-Left_Hidden projection')

        # Create a plastic connection between Ball Y and Hidden neurons
        p.Projection(
            ball_y_pop, left_hidden_pop,
            p.FromListConnector(prev_ball_y_left_conn)
            if load_previous_connections else p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Ball_y-Left_Hidden projection')

        # Create a plastic connection between Paddle and Hidden neurons
        p.Projection(
            self.paddle_pop, left_hidden_pop,
            p.FromListConnector(prev_paddle_left_conn)
            if load_previous_connections else p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Paddle-Left_Hidden projection')

        # Create Dopaminergic connections
        p.Projection(
            ball_on_left_dopaminergic_pop, left_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=p.extra_models.Neuromodulation(
                weight=dopaminergic_weight, tau_c=30., tau_d=10.),
            receptor_type='reward',
            label='reward ball on left synapses -> left hidden')
        p.Projection(
            ball_on_right_dopaminergic_pop, left_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=p.extra_models.Neuromodulation(
                weight=dopaminergic_weight, tau_c=30., tau_d=10.),
            receptor_type='punishment',
            label='punish ball on right synapses -> left hidden')

        # --------------------------------------------------------------------
        # Right Hidden Population
        # --------------------------------------------------------------------

        right_hidden_pop = p.Population(
            hidden_pop_size, p.IF_curr_exp(),
            label="right_hidden_pop")

        # Stimulate Right Hidden pop
        p.Projection(
            stimulation_pop, right_hidden_pop,
            p.OneToOneConnector(),
            p.StaticSynapse(weight=stim_weight))

        if load_previous_connections:
            prev_ball_x_right_conn = previous_connections[3]
            prev_ball_y_right_conn = previous_connections[4]
            prev_paddle_right_conn = previous_connections[5]

        # Create a plastic connection between Ball X and Hidden neurons
        self.ball_x_learning_proj = p.Projection(
            ball_x_pop, right_hidden_pop,
            p.FromListConnector(prev_ball_x_right_conn)
            if load_previous_connections else p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Ball_x-Right_Hidden projection')

        # Create a plastic connection between Ball Y and Hidden neurons
        p.Projection(
            ball_y_pop, right_hidden_pop,
            p.FromListConnector(prev_ball_y_right_conn)
            if load_previous_connections else p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Ball_y-Right_Hidden projection')

        # Create a plastic connection between Paddle and Hidden neurons
        p.Projection(
            self.paddle_pop, right_hidden_pop,
            p.FromListConnector(prev_paddle_right_conn)
            if load_previous_connections else p.AllToAllConnector(),
            synapse_type=hidden_synapse_dynamics,
            receptor_type='excitatory', label='Paddle-Right_Hidden projection')

        # Create Dopaminergic connections
        p.Projection(
            ball_on_left_dopaminergic_pop, right_hidden_pop,
            p.AllToAllConnector(),
            synapse_type=p.extra_models.Neuromodulation(
                weight=dopaminergic_weight, tau_c=30., tau_d=10.),
            receptor_type='punishment',
            label='punish ball on left synapses -> right hidden')
        p.Projection(
            ball_on_right_dopaminergic_pop, right_hidden_pop,
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

        decision_input_pop = p.Population(
            2, p.IF_cond_exp, label="decision_input_pop")

        [left_decision_conn, right_decision_conn] = \
            get_hidden_to_decision_connections(
                hidden_pop_size, weight=hidden_to_decision_weight)

        p.Projection(
            left_hidden_pop, decision_input_pop,
            p.FromListConnector(left_decision_conn),
            p.StaticSynapse(weight=hidden_to_decision_weight))
        p.Projection(
            right_hidden_pop, decision_input_pop,
            p.FromListConnector(right_decision_conn),
            p.StaticSynapse(weight=hidden_to_decision_weight))

        # Connect input decision population to the game
        p.Projection(
            decision_input_pop, self.breakout_pop, p.OneToOneConnector(),
            p.StaticSynapse(weight=1.))
