"""
IF_curr_exp_izhikevich_neuromodulation neuron firing experiment
"""
from pyNN.utility.plotting import Figure, Panel
import spynnaker8 as sim


weight = 4.8
simtime = 500

cell_params = {
    'cm': 1.0,          # nF membrane capacitance
    'i_offset': 0.0,    # nA    bias current
    'tau_m': 20.0,      # ms    membrane time constant
    'tau_refrac': 0.1,  # ms    refractory period
    'tau_syn_E': 5.0,   # ms    excitatory synapse time constant
    'tau_syn_I': 5.0,   # ms    inhibitory synapse time constant
    'v_reset': -65.0,   # mV    reset membrane potential
    'v_rest': -65.0,    # mV    rest membrane potential
    'v_thresh': -50.0,  # mV    firing threshold voltage
}

sim.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)


spikeArray = {'spike_times': [[simtime // 2]]}
stimulus = sim.Population(1, sim.SpikeSourceArray, spikeArray, label='stimulus')
stimulus.record("spikes")

curr_izh_pop = sim.Population(1, sim.IF_curr_exp_izhikevich_neuromodulation, cell_params,
                              label='IF_curr_exp_izhikevich_neuromodulation neuron population')
curr_izh_pop.record(["spikes", "v"])

synapse_dynamics = sim.STDPMechanism(
    timing_dependence=sim.IzhikevichNeuromodulation(
        tau_plus=15., tau_minus=5.,
        A_plus=0.1, A_minus=0.1,
        tau_c=150., tau_d=10.),
    weight_dependence=sim.MultiplicativeWeightDependence(w_min=0, w_max=weight),
    weight=weight,
    neuromodulation=True)
sim.Projection(stimulus, curr_izh_pop, sim.AllToAllConnector(),
               synapse_type=synapse_dynamics)

sim.run(simtime)

stimulus_spikes = stimulus.get_data("spikes")
curr_izh_neo = curr_izh_pop.get_data(variables=["spikes", "v"])

sim.end()

Figure(
    Panel(stimulus_spikes.segments[0].spiketrains,
          yticks=True, xticks=True, markersize=2, xlim=(0, simtime)),
    Panel(curr_izh_neo.segments[0].spiketrains,
          yticks=True, xticks=True, markersize=2, xlim=(0, simtime)),
    Panel(curr_izh_neo.segments[0].filter(name='v')[0], ylabel="Membrane potential (mV)",
          data_labels=[curr_izh_pop.label], yticks=True, xticks=True,
          ylim=(-70, -40),  xlim=(0, simtime)),
    title="IF_curr_exp_izhikevich_neuromodulation neuron firing experiment"
)



