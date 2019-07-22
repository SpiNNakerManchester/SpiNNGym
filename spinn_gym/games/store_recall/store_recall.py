# PACMAN imports
# from spynnaker.pyNN.models.common.population_settable_change_requires_mapping import \
#     PopulationSettableChangeRequiresMapping

# from spynnaker.pyNN.models.abstract_models import AbstractPopulationSettable
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun

from pacman.executor.injection_decorator import inject_items
from pacman.model.constraints.key_allocator_constraints import ContiguousKeyRangeContraint
from pacman.model.decorators.overrides import overrides
from pacman.model.graphs.application import ApplicationVertex
from pacman.model.resources.cpu_cycles_per_tick_resource import \
    CPUCyclesPerTickResource
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.sdram_resource import SDRAMResource
from pacman.model.resources.variable_sdram import VariableSDRAM

from spinn_front_end_common.interface.buffer_management \
    import recording_utilities

# SpinnFrontEndCommon imports
# from spinn_front_end_common.abstract_models \
#     .abstract_binary_uses_simulation_run import AbstractBinaryUsesSimulationRun
from spinn_front_end_common.abstract_models \
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models. \
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.utilities import globals_variables

from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants

from spinn_front_end_common.utilities.utility_objs import ExecutableType

# from spinn_front_end_common.utilities.utility_objs.executable_start_type \
#     import ExecutableStartType

from spinn_front_end_common.utilities import globals_variables

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
from spynnaker.pyNN.models.common import NeuronRecorder
from spynnaker.pyNN.models.neuron import AbstractPopulationVertex
from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

from spinn_front_end_common.abstract_models \
    .abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition

# Recall imports
from store_recall_machine_vertex import RecallMachineVertex

import numpy

from data_specification.enums.data_type import DataType

NUMPY_DATA_ELEMENT_TYPE = numpy.double


# ----------------------------------------------------------------------------
# Recall
# ----------------------------------------------------------------------------
# **HACK** for Projection to connect a synapse type is required
# class RecallSynapseType(object):
#     def get_synapse_id_by_target(self, target):
#         return 0


class Bad_Table(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# ----------------------------------------------------------------------------
# Recall
# ----------------------------------------------------------------------------
class Recall(ApplicationVertex,
             AbstractGeneratesDataSpecification,
             AbstractHasAssociatedBinary,
             AbstractProvidesOutgoingPartitionConstraints,
             AbstractAcceptsIncomingSynapses,
             AbstractNeuronRecordable,
             SimplePopulationSettable,
             AbstractProvidesNKeysForPartition
             # AbstractBinaryUsesSimulationRun
             ):

    def get_connections_from_machine(self, transceiver, placement, edge, graph_mapper,
                                     routing_infos, synapse_information, machine_time_step):

        super(Recall, self).get_connections_from_machine(transceiver, placement, edge,
                                                        graph_mapper, routing_infos,
                                                        synapse_information,
                                                        machine_time_step)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def add_pre_run_connection_holder(self, connection_holder, projection_edge, synapse_information):
        super(Recall, self).add_pre_run_connection_holder(connection_holder, projection_edge, synapse_information)

    # def get_binary_start_type(self):
    #     super(Recall, self).get_binary_start_type()
    #
    # def requires_mapping(self):
    #     pass

    def clear_connection_cache(self):
        pass

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return self._n_neurons  # 2  # two for control IDs

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    RECALL_REGION_BYTES = 200
    DATA_REGION_BYTES = 200
    MAX_SIM_DURATION = 1000 * 60 * 60 * 24  # 1 day

    # parameters expected by PyNN
    default_parameters = {
        'time_period': 200.0,
        'constraints': None,
        'rate_on': 50.0,
        'rate_off': 0.0,
        'pop_size': 1,
        'prob_command': 1.0/6.0,
        'prob_in_change': 1.0/2.0,
        'stochastic': 1,
        'reward': 0,
        'label': "Recall",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION,
        'random_seed': [numpy.random.randint(10000), numpy.random.randint(10000),
                        numpy.random.randint(10000), numpy.random.randint(10000)]}

    # **HACK** for Projection to connect a synapse type is required
    # synapse_type = RecallSynapseType()

    def __init__(self,
                 rate_on=default_parameters['rate_on'],
                 rate_off=default_parameters['rate_off'],
                 pop_size=default_parameters['pop_size'],
                 prob_command=default_parameters['prob_command'],
                 prob_in_change=default_parameters['prob_in_change'],
                 time_period=default_parameters['time_period'],
                 stochastic=default_parameters['stochastic'],
                 reward=default_parameters['reward'],
                 constraints=default_parameters['constraints'],
                 label=default_parameters['label'],
                 incoming_spike_buffer_size=default_parameters['incoming_spike_buffer_size'],
                 simulation_duration_ms=default_parameters['duration'],
                 rand_seed=default_parameters['random_seed']):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label

        # Pass in variables
        self._rate_on = rate_on
        self._rate_off = rate_off
        self._stochastic = stochastic
        self._reward = reward
        self._pop_size = pop_size
        self._prob_command = prob_command
        self._prob_in_change = prob_in_change

        self._n_neurons = pop_size * 4
        self._rand_seed = rand_seed

        self._time_period = time_period

        # used to define size of recording region
        self._recording_size = int((simulation_duration_ms / 1000.) * 4)

        # Superclasses
        ApplicationVertex.__init__(
            self, label, constraints, self.n_atoms)
        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        SimplePopulationSettable.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractAcceptsIncomingSynapses.__init__(self)
        self._change_requires_mapping = True
        # get config from simulator
        config = globals_variables.get_simulator().config

        if incoming_spike_buffer_size is None:
            self._incoming_spike_buffer_size = config.getint(
                "Simulation", "incoming_spike_buffer_size")

        # PopulationSettableChangeRequiresMapping.__init__(self)
        # self.width = width
        # self.height = height

    def neurons(self):
        return self._n_neurons

    def get_maximum_delay_supported_in_ms(self, machine_time_step):
        # Recall has no synapses so can simulate only one time step of delay
        return machine_time_step / 1000.0

    #    def get_max_atoms_per_core(self):
    #       return self.n_atoms

    # ------------------------------------------------------------------------
    # ApplicationVertex overrides
    # ------------------------------------------------------------------------
    @overrides(ApplicationVertex.get_resources_used_by_atoms)
    def get_resources_used_by_atoms(self, vertex_slice):
        # **HACK** only way to force no partitioning is to zero dtcm and cpu
        container = ResourceContainer(
            # sdram=SDRAMResource(
            #     self.RECALL_REGION_BYTES +
            #     front_end_common_constants.SYSTEM_BYTES_REQUIREMENT),
            sdram=VariableSDRAM(fixed_sdram=0, per_timestep_sdram=8),
            dtcm=DTCMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0))

        return container

    @overrides(ApplicationVertex.create_machine_vertex)
    def create_machine_vertex(self, vertex_slice, resources_required,
                              label=None, constraints=None):
        # Return suitable machine vertex
        return RecallMachineVertex(resources_required, constraints, self._label)

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self):
        return self._n_neurons

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"machine_time_step": "MachineTimeStep",
                   "time_scale_factor": "TimeScaleFactor",
                   "graph_mapper": "MemoryGraphMapper",
                   "routing_info": "MemoryRoutingInfos",
                   "tags": "MemoryTags",
                   "n_machine_time_steps": "DataNTimeSteps"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"machine_time_step", "time_scale_factor",
                                     "graph_mapper", "routing_info", "tags",
                                     "n_machine_time_steps"}
               )
    def generate_data_specification(self, spec, placement, machine_time_step,
                                    time_scale_factor, graph_mapper,
                                    routing_info, tags, n_machine_time_steps):
        vertex = placement.vertex
        vertex_slice = graph_mapper.get_slice(vertex)

        spec.comment("\n*** Spec for Recall Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=RecallMachineVertex._RECALL_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=RecallMachineVertex._RECALL_REGIONS.RECALL.value,
            size=self.RECALL_REGION_BYTES, label='RecallParams')
        # vertex.reserve_provenance_data_region(spec)
        # reserve recording region
        spec.reserve_memory_region(
            RecallMachineVertex._RECALL_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=RecallMachineVertex._RECALL_REGIONS.DATA.value,
            size=self.DATA_REGION_BYTES, label='RecallArms')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            RecallMachineVertex._RECALL_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # Write recall region containing routing key to transmit with
        spec.comment("\nWriting recall region:\n")
        spec.switch_write_focus(
            RecallMachineVertex._RECALL_REGIONS.RECALL.value)
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.SPIKE_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting recall recording region:\n")
        spec.switch_write_focus(
            RecallMachineVertex._RECALL_REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size], ip_tags=ip_tags))

        # Write probabilites for arms
        spec.comment("\nWriting arm probability region region:\n")
        spec.switch_write_focus(
            RecallMachineVertex._RECALL_REGIONS.DATA.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_value(self._time_period, data_type=DataType.UINT32)
        spec.write_value(self._pop_size, data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[3], data_type=DataType.UINT32)
        spec.write_value(self._rate_on, data_type=DataType.UINT32)
        spec.write_value(self._rate_off, data_type=DataType.UINT32)
        spec.write_value(self._stochastic, data_type=DataType.UINT32)
        spec.write_value(self._reward, data_type=DataType.UINT32)
        spec.write_value(self._prob_command, data_type=DataType.S1615)
        spec.write_value(self._prob_in_change, data_type=DataType.S1615)

        # End-of-Spec:
        spec.end_specification()

    # ------------------------------------------------------------------------
    # AbstractHasAssociatedBinary overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "store_recall.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        # return ExecutableStartType.USES_SIMULATION_INTERFACE
        return ExecutableType.USES_SIMULATION_INTERFACE

    # ------------------------------------------------------------------------
    # AbstractProvidesOutgoingPartitionConstraints overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractProvidesOutgoingPartitionConstraints.
               get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return [ContiguousKeyRangeContraint()]

    @property
    @overrides(AbstractChangableAfterRun.requires_mapping)
    def requires_mapping(self):
        return self._change_requires_mapping

    @overrides(AbstractChangableAfterRun.mark_no_changes)
    def mark_no_changes(self):
        self._change_requires_mapping = False

    @overrides(SimplePopulationSettable.set_value)
    def set_value(self, key, value):
        SimplePopulationSettable.set_value(self, key, value)
        self._change_requires_neuron_parameters_reload = True

    # ------------------------------------------------------------------------
    # Recording overrides
    # ------------------------------------------------------------------------
    @overrides(
        AbstractNeuronRecordable.clear_recording)
    def clear_recording(
            self, variable, buffer_manager, placements, graph_mapper):
        self._clear_recording_region(
            buffer_manager, placements, graph_mapper,
            0)

    @overrides(AbstractNeuronRecordable.get_recordable_variables)
    def get_recordable_variables(self):
        return 'score'

    @overrides(AbstractNeuronRecordable.is_recording)
    def is_recording(self, variable):
        return True

    @overrides(AbstractNeuronRecordable.set_recording)
    def set_recording(self, variable, new_state=True, sampling_interval=None,
                      indexes=None):
        a = 1

    @overrides(AbstractNeuronRecordable.get_neuron_sampling_interval)
    def get_neuron_sampling_interval(self, variable):
        return 10000  # 10 seconds hard coded in bkout.c

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(self, variable, n_machine_time_steps, placements,
                 graph_mapper, buffer_manager, machine_time_step):
        vertex = graph_mapper.get_machine_vertices(self).pop()
        placement = placements.get_placement_of_vertex(vertex)

        # Read the data recorded
        # data_values, _ = buffer_manager.get_data_for_vertex(placement, 0)
        data_values, _ = buffer_manager.get_data_by_placement(placement,0)
        data = data_values#.read_all()

        numpy_format = list()
        numpy_format.append(("Score", numpy.int32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

        # return formatted_data
        return output_data

