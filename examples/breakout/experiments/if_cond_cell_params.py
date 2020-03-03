"""
IF_cond_exp neuron firing experiment
"""
from pyNN.utility.plotting import Figure, Panel
import spynnaker8 as sim


weight = 0.085 * 200
simtime = 100

# Findings:
#   --> multiply cm by 200 to scale up a neuron

cell_params = {
    'cm': 1.0 * 200,    # nF membrane capacitance
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

spikeArray = {'spike_times': [[simtime // 4]]}
stimulus = sim.Population(1, sim.SpikeSourceArray, spikeArray, label='stimulus')
stimulus.record("spikes")

cond_pop = sim.Population(1, sim.IF_cond_exp, cell_params, label='IF_cond_exp neuron population')
cond_pop.record(["spikes", "v"])

sim.Projection(stimulus, cond_pop, sim.AllToAllConnector(),
               synapse_type=sim.StaticSynapse(weight=weight))

sim.run(simtime)

stimulus_spikes = stimulus.get_data("spikes")
cond_neo = cond_pop.get_data(variables=["spikes", "v"])

sim.end()

Figure(
    Panel(stimulus_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, simtime)),
    Panel(cond_neo.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, simtime)),
    Panel(cond_neo.segments[0].filter(name='v')[0], ylabel="Membrane potential (mV)",
          data_labels=[cond_pop.label], yticks=True, xlim=(0, simtime)),
    title="IF_cond_exp neuron firing experiment"
)



