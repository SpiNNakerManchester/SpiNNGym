import numpy
import math

from spinn_utilities.overrides import overrides

# PACMAN imports
# from pacman.executor.injection_decorator import inject_items
from pacman.model.constraints.key_allocator_constraints import \
    ContiguousKeyRangeContraint
# from pacman.model.graphs.application import ApplicationVertex
# from pacman.model.resources.cpu_cycles_per_tick_resource import \
#     CPUCyclesPerTickResource
# from pacman.model.resources.dtcm_resource import DTCMResource
# from pacman.model.resources.resource_container import ResourceContainer
# from pacman.model.resources.variable_sdram import VariableSDRAM
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)
from pacman.model.graphs.common import Slice

# SpinnFrontEndCommon imports
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun
# from spinn_front_end_common.interface.buffer_management \
#     import recording_utilities
# from spinn_front_end_common.abstract_models \
#     .abstract_generates_data_specification \
#     import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models. \
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.utilities import globals_variables
# from spinn_front_end_common.interface.simulation import simulation_utilities
# from spinn_front_end_common.utilities import constants as \
#     front_end_common_constants

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import \
    AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
# from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

# Pendulum imports
from spinn_gym.games.double_inverted_pendulum.double_pendulum_machine_vertex \
    import DoublePendulumMachineVertex

NUMPY_DATA_ELEMENT_TYPE = numpy.double


# ----------------------------------------------------------------------------
# Double Pendulum
# ----------------------------------------------------------------------------
class DoublePendulum(AbstractOneAppOneMachineVertex,
                     AbstractProvidesOutgoingPartitionConstraints,
                     AbstractAcceptsIncomingSynapses, AbstractNeuronRecordable,
                     SimplePopulationSettable):

    @overrides(AbstractAcceptsIncomingSynapses.verify_splitter)
    def verify_splitter(self, splitter):
        # Need to ignore this verify
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(
            self, transceiver, placements, app_edge, synapse_info):

        # TODO: make this work properly (the following call does nothing)

        super(DoublePendulum, self).get_connections_from_machine(
            transceiver, placements, app_edge, synapse_info)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def clear_connection_cache(self):
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    PENDULUM_REGION_BYTES = 4
    BASE_DATA_REGION_BYTES = 17 * 4
    MAX_SIM_DURATION = 1000 * 60 * 60 * 24 * 7  # 1 week

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
                 incoming_spike_buffer_size=default_parameters[
                     'incoming_spike_buffer_size'],
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

        self._force_increments = force_increments
        # for rate based it's only 1 neuron per metric
        # (position, angle, velocity of both)
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

        # technically as using OneAppOneMachine this is not necessary?
        resources_required = (
            self.PENDULUM_REGION_BYTES + self.BASE_DATA_REGION_BYTES +
            self._recording_size)

        vertex_slice = Slice(0, self._n_neurons-1)

        # Superclasses
        super(DoublePendulum, self).__init__(
            DoublePendulumMachineVertex(
                vertex_slice, resources_required, constraints, label, self,
                encoding, time_increment, pole_length, pole_angle,
                pole2_length, pole2_angle, reward_based, force_increments,
                max_firing_rate, number_of_bins, central, bin_overlap,
                tau_force, incoming_spike_buffer_size, simulation_duration_ms,
                rand_seed),
            label=label, constraints=constraints)
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

    def neurons(self):
        return self._n_neurons

    def get_maximum_delay_supported_in_ms(self, machine_time_step):
        # Pendulum has no synapses so can simulate only one time step of delay
        return machine_time_step / 1000.0

    # ------------------------------------------------------------------------
    # ApplicationVertex overrides
    # ------------------------------------------------------------------------
#     @overrides(ApplicationVertex.get_resources_used_by_atoms)
#     def get_resources_used_by_atoms(self, vertex_slice):
#         # **HACK** only way to force no partitioning is to zero dtcm and cpu
#         container = ResourceContainer(
#             # sdram=SDRAMResource(
#             #     self.PENDULUM_REGION_BYTES +
#             #     front_end_common_constants.SYSTEM_BYTES_REQUIREMENT),
#             sdram=VariableSDRAM(fixed_sdram=0, per_timestep_sdram=12),
#             dtcm=DTCMResource(0),
#             cpu_cycles=CPUCyclesPerTickResource(0))
#
#         return container
#
#     @overrides(ApplicationVertex.create_machine_vertex)
#     def create_machine_vertex(self, vertex_slice, resources_required,
#                               label=None, constraints=None):
#         # Return suitable machine vertex
#         return DoublePendulumMachineVertex(
#             resources_required, constraints, self._label, self, vertex_slice)

    @property
    @overrides(AbstractOneAppOneMachineVertex.n_atoms)
    def n_atoms(self):
        return self._n_neurons

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
#     @inject_items({"machine_time_step": "MachineTimeStep",
#                    "time_scale_factor": "TimeScaleFactor",
#                    "routing_info": "MemoryRoutingInfos",
#                    "tags": "MemoryTags"})
#     @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
#                additional_arguments={"machine_time_step", "time_scale_factor",
#                                      "routing_info", "tags"}
#                )
#     def generate_data_specification(self, spec, placement, machine_time_step,
#                                     time_scale_factor, routing_info, tags):
#         vertex = placement.vertex
#
#         spec.comment("\n*** Spec for Double Pendulum Instance ***\n\n")
#         spec.comment("\nReserving memory space for data regions:\n\n")
#
#         # Reserve memory:
#         spec.reserve_memory_region(
#             region=DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS
#             .SYSTEM.value,
#             size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
#             label='setup')
#         spec.reserve_memory_region(
#             region=DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS
#             .PENDULUM.value,
#             size=self.PENDULUM_REGION_BYTES, label='PendulumVertex')
#         # reserve recording region
#         spec.reserve_memory_region(
#             DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS
#             .RECORDING.value,
#             recording_utilities.get_recording_header_size(1))
#         spec.reserve_memory_region(
#             region=DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS
#             .DATA.value,
#             size=self.DATA_REGION_BYTES, label='PendulumData')
#
#         # Write setup region
#         spec.comment("\nWriting setup region:\n")
#         spec.switch_write_focus(
#             DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS
#             .SYSTEM.value)
#         spec.write_array(simulation_utilities.get_simulation_header_array(
#             vertex.get_binary_file_name(), machine_time_step,
#             time_scale_factor))
#
#         # Write pendulum region containing routing key to transmit with
#         spec.comment("\nWriting double pendulum region:\n")
#         spec.switch_write_focus(
#             DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS
#             .PENDULUM.value)
#         spec.write_value(routing_info.get_first_key_from_pre_vertex(
#             vertex, constants.SPIKE_PARTITION_ID))
#
#         # Write recording region for score
#         spec.comment("\nWriting double pendulum recording region:\n")
#         spec.switch_write_focus(
#             DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS
#             .RECORDING.value)
#         ip_tags = tags.get_ip_tags_for_vertex(self) or []
#         spec.write_array(recording_utilities.get_recording_header_array(
#             [self._recording_size], ip_tags=ip_tags))
#
#         # Write probabilites for arms
#         spec.comment("\nWriting double pendulum data region:\n")
#         spec.switch_write_focus(
#             DoublePendulumMachineVertex._DOUBLE_PENDULUM_REGIONS.DATA.value)
#         spec.write_value(self._encoding, data_type=DataType.UINT32)
#         spec.write_value(self._time_increment, data_type=DataType.UINT32)
#         spec.write_value(self._pole_length, data_type=DataType.S1615)
#         spec.write_value(self._pole_angle, data_type=DataType.S1615)
#         spec.write_value(self._pole2_length, data_type=DataType.S1615)
#         spec.write_value(self._pole2_angle, data_type=DataType.S1615)
#         spec.write_value(self._reward_based, data_type=DataType.UINT32)
#         spec.write_value(self._force_increments, data_type=DataType.UINT32)
#         spec.write_value(self._max_firing_rate, data_type=DataType.UINT32)
#         spec.write_value(self._number_of_bins, data_type=DataType.UINT32)
#         spec.write_value(self._central, data_type=DataType.UINT32)
#         spec.write_value(self._rand_seed[0], data_type=DataType.UINT32)
#         spec.write_value(self._rand_seed[1], data_type=DataType.UINT32)
#         spec.write_value(self._rand_seed[2], data_type=DataType.UINT32)
#         spec.write_value(self._rand_seed[3], data_type=DataType.UINT32)
#         spec.write_value(self._bin_overlap, data_type=DataType.S1615)
#         spec.write_value(self._tau_force, data_type=DataType.S1615)
#
#         # End-of-Spec:
#         spec.end_specification()

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
            self, variable, buffer_manager, placements):
        self._clear_recording_region(buffer_manager, placements, 0)

    @overrides(AbstractNeuronRecordable.get_recordable_variables)
    def get_recordable_variables(self):
        return 'score'

    @overrides(AbstractNeuronRecordable.is_recording)
    def is_recording(self, variable):
        return True

    @overrides(AbstractNeuronRecordable.set_recording)
    def set_recording(self, variable, new_state=True, sampling_interval=None,
                      indexes=None):
        pass

    @overrides(AbstractNeuronRecordable.get_expected_n_rows)
    def get_expected_n_rows(
            self, n_machine_time_steps, sampling_rate, vertex, variable):
        # Just copying what's in NeuronRecorder for now...
        return int(math.ceil(n_machine_time_steps / sampling_rate))

    @overrides(AbstractNeuronRecordable.get_neuron_sampling_interval)
    def get_neuron_sampling_interval(self, variable):
        return 10000  # 10 seconds hard coded in bkout.c

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(self, variable, n_machine_time_steps, placements,
                 buffer_manager, machine_time_step):
        vertex = self.machine_vertices.pop()
        placement = placements.get_placement_of_vertex(vertex)

        # Read the data recorded
        data_values, _ = buffer_manager.get_data_by_placement(placement, 0)
        data = data_values

        numpy_format = list()
        numpy_format.append(("Score", numpy.float32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

        # return formatted_data
        return output_data

    def _clear_recording_region(
            self, buffer_manager, placements, recording_region_id):
        """ Clear a recorded data region from the buffer manager.

        :param buffer_manager: the buffer manager object
        :param placements: the placements object
        :param recording_region_id: the recorded region ID for clearing
        :rtype: None
        """
        for machine_vertex in self.machine_vertices:
            placement = placements.get_placement_of_vertex(machine_vertex)
            buffer_manager.clear_recorded_data(
                placement.x, placement.y, placement.p, recording_region_id)

    def reset_ring_buffer_shifts(self):
        pass

    def __str__(self):
        return "{} with {} atoms".format(self._label, self.n_atoms)

    def __repr__(self):
        return self.__str__()
