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

from spinn_front_end_common.abstract_models\
   .abstract_provides_n_keys_for_partition \
   import AbstractProvidesNKeysForPartition

# Pendulum imports
from double_pendulum_vertex import DoublePendulumMachineVertex

import numpy

from data_specification.enums.data_type import DataType

NUMPY_DATA_ELEMENT_TYPE = numpy.double


# ----------------------------------------------------------------------------
# Double Pendulum
# ----------------------------------------------------------------------------
class DoublePendulum(ApplicationVertex,
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

        super(DoublePendulum, self).get_connections_from_machine(transceiver, placement, edge,
                                                           graph_mapper, routing_infos,
                                                           synapse_information,
                                                           machine_time_step)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def add_pre_run_connection_holder(self, connection_holder, projection_edge, synapse_information):
        super(DoublePendulum, self).add_pre_run_connection_holder(connection_holder, projection_edge, synapse_information)

    def clear_connection_cache(self):
        pass

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        return self._n_neurons  # 2  # two for control IDs

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    PENDULUM_REGION_BYTES = 24
    DATA_REGION_BYTES = 80
    MAX_SIM_DURATION = 1000 * 60 * 60 * 24 * 7 # 1 week

    # parameters expected by PyNN
    default_parameters = {
        'constraints': None,
        'encoding': 0,  # 0 rate, 1 receptive bins, 2 spike time, 3 rank
        'time_increment': 20,
        'pole_length': 1.0,
        'pole_angle': 0.1,
        'pole2_length': 0.1,
        'pole2_angle': 0,
        'reward_based': 1,
        'force_increments': 100,
        'max_firing_rate': 100,
        'number_of_bins': 20,
        'central': 1,
        'rand_seed': [0, 1, 2, 3],
        'bin_overlap': 2,
        'tau_force': 0,
        'label': "pole",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION}

    # **HACK** for Projection to connect a synapse type is required

    def __init__(self, constraints=default_parameters['constraints'],
                 encoding=default_parameters['encoding'],
                 time_increment=default_parameters['time_increment'],
                 pole_length=default_parameters['pole_length'],
                 pole_angle=default_parameters['pole_angle'],
                 pole2_length=default_parameters['pole2_length'],
                 pole2_angle=default_parameters['pole2_angle'],
                 reward_based=default_parameters['reward_based'],
                 force_increments=default_parameters['force_increments'],
                 max_firing_rate=default_parameters['max_firing_rate'],
                 number_of_bins=default_parameters['number_of_bins'],
                 central=default_parameters['central'],
                 rand_seed=default_parameters['rand_seed'],
                 bin_overlap=default_parameters['bin_overlap'],
                 tau_force=default_parameters['tau_force'],
                 label=default_parameters['label'],
                 incoming_spike_buffer_size=default_parameters['incoming_spike_buffer_size'],
                 simulation_duration_ms=default_parameters['duration']):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label

        self._encoding = encoding

        # Pass in variables
        self._pole_length = pole_length
        self._pole_angle = pole_angle
        self._pole2_length = pole2_length
        self._pole2_angle = pole2_angle

        self._force_increments= force_increments
        # for rate based it's only 1 neuron per metric (position, angle, velocity of both)
        self._n_neurons = 6 * number_of_bins

        self._time_increment = time_increment
        self._reward_based = reward_based

        self._max_firing_rate = max_firing_rate
        self._number_of_bins = number_of_bins
        self._central = central
        self._rand_seed = rand_seed
        self._bin_overlap = bin_overlap
        self._tau_force = tau_force

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
        # Pendulum has no synapses so can simulate only one time step of delay
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
            sdram=SDRAMResource(
                self.PENDULUM_REGION_BYTES +
                front_end_common_constants.SYSTEM_BYTES_REQUIREMENT),
            dtcm=DTCMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0))

        return container

    @overrides(ApplicationVertex.create_machine_vertex)
    def create_machine_vertex(self, vertex_slice, resources_required,
                              label=None, constraints=None):
        # Return suitable machine vertex
        return DoublePendulumMachineVertex(resources_required, constraints, self._label)

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
                   "n_machine_time_steps": "TotalMachineTimeSteps"})
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

        spec.comment("\n*** Spec for Double Pendulum Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.PENDULUM.value,
            size=self.PENDULUM_REGION_BYTES, label='PendulumVertex')
        # vertex.reserve_provenance_data_region(spec)
        # reserve recording region
        spec.reserve_memory_region(
            DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.DATA.value,
            size=self.DATA_REGION_BYTES, label='PendulumData')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # Write pendulum region containing routing key to transmit with
        spec.comment("\nWriting double pendulum region:\n")
        spec.switch_write_focus(
            DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.PENDULUM.value)
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.SPIKE_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting double pendulum recording region:\n")
        spec.switch_write_focus(
            DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size], ip_tags=ip_tags))

        # Write probabilites for arms
        spec.comment("\nWriting arm probability region region:\n")
        spec.switch_write_focus(
            DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.DATA.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_value(self._encoding, data_type=DataType.UINT32)
        spec.write_value(self._time_increment, data_type=DataType.UINT32)
        spec.write_value(self._pole_length, data_type=DataType.S1615)
        spec.write_value(self._pole_angle, data_type=DataType.S1615)
        spec.write_value(self._pole2_length, data_type=DataType.S1615)
        spec.write_value(self._pole2_angle, data_type=DataType.S1615)
        spec.write_value(self._reward_based, data_type=DataType.UINT32)
        spec.write_value(self._force_increments, data_type=DataType.UINT32)
        spec.write_value(self._max_firing_rate, data_type=DataType.UINT32)
        spec.write_value(self._number_of_bins, data_type=DataType.UINT32)
        spec.write_value(self._central, data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[3], data_type=DataType.UINT32)
        spec.write_value(self._bin_overlap, data_type=DataType.S1615)
        spec.write_value(self._tau_force, data_type=DataType.S1615)

        # End-of-Spec:
        spec.end_specification()

    # ------------------------------------------------------------------------
    # AbstractHasAssociatedBinary overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "double_pendulum.aplx"

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
        data_values, _ = buffer_manager.get_data_for_vertex(placement, 0)
        data = data_values.read_all()

        numpy_format = list()
        numpy_format.append(("Score", numpy.float32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

        # return formatted_data
        return output_data

