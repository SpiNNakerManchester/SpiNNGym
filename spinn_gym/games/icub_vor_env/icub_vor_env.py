from spinn_utilities.overrides import overrides

# PACMAN imports
from pacman.executor.injection_decorator import inject_items
from pacman.model.constraints.key_allocator_constraints import \
    ContiguousKeyRangeContraint
from pacman.model.graphs.application import ApplicationVertex
from pacman.model.resources.cpu_cycles_per_tick_resource import \
    CPUCyclesPerTickResource
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.variable_sdram import VariableSDRAM

from data_specification.enums.data_type import DataType

# SpinnFrontEndCommon imports
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.abstract_models \
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models. \
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.utilities import globals_variables
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants
from spinn_front_end_common.utilities.exceptions import ConfigurationException

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import \
    AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

# ICubVorEnv imports
from spinn_gym.games.icub_vor_env.icub_vor_env_machine_vertex \
    import ICubVorEnvMachineVertex

import numpy

NUMPY_DATA_ELEMENT_TYPE = numpy.double


# ----------------------------------------------------------------------------
# ICubVorEnv
# ----------------------------------------------------------------------------
class ICubVorEnv(ApplicationVertex, AbstractGeneratesDataSpecification,
                 AbstractProvidesOutgoingPartitionConstraints,
                 AbstractAcceptsIncomingSynapses, AbstractNeuronRecordable,
                 SimplePopulationSettable):

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(
            self, transceiver, placements, app_edge, synapse_info):

        # TODO: make this work properly (the following call does nothing)

        super(ICubVorEnv, self).get_connections_from_machine(
            transceiver, placements, app_edge, synapse_info)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def clear_connection_cache(self):
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        return 0

    # key value
    ICUB_VOR_ENV_REGION_BYTES = 4
    # error_window_size, output_size, number_of_inputs
    BASE_DATA_REGION_BYTES = 5 * 4
    # not sure this is entirely necessary but keeping it for now
    MAX_SIM_DURATION = 10000
    # Probably better ways of doing this too, but keeping it for now
    RECORDABLE_VARIABLES = [
        "l_count", "r_count", "error", "eye_pos", "eye_vel"]
    RECORDABLE_DTYPES = [
        DataType.UINT32, DataType.UINT32, DataType.S1615, DataType.S1615,
        DataType.S1615]

    # parameters expected by PyNN
    default_parameters = {
        'error_window_size': 10,
        'gain': 20,
        'pos_to_vel': 1 / (0.001 * 2 * numpy.pi * 10),
        'output_size': 200,
        'constraints': None,
        'label': "ICubVorEnv",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION}

    def __init__(self, head_pos, head_vel, perfect_eye_pos, perfect_eye_vel,
                 error_window_size=default_parameters['error_window_size'],
                 output_size=default_parameters['output_size'],
                 gain=default_parameters['gain'],
                 pos_to_vel=default_parameters['pos_to_vel'],
                 constraints=default_parameters['constraints'],
                 label=default_parameters['label'],
                 incoming_spike_buffer_size=default_parameters[
                     'incoming_spike_buffer_size'],
                 simulation_duration_ms=default_parameters['duration']):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label

        # Pass in variables
        self._head_pos = head_pos
        self._head_vel = head_vel
        self._perfect_eye_pos = perfect_eye_pos
        self._perfect_eye_vel = perfect_eye_vel
        self._error_window_size = error_window_size
        self._output_size = output_size
        self._gain = gain
        self._pos_to_vel = pos_to_vel
        self._number_of_inputs = len(head_pos)
        if self._number_of_inputs != len(head_vel):
            raise ConfigurationException(
                "The length of head_positions {} is not the same as the "
                "length of head_velocities {}".format(
                    self._number_of_inputs, len(head_vel)))

        # n_neurons is the number of atoms in the network, which in this
        # case only needs to be 2 (for receiving "left" and "right")
        self._n_neurons = 2

        # used to define size of recording region:
        # record variables every error_window_size ms (same size each time)
        self._n_recordable_variables = len(self.RECORDABLE_VARIABLES)

        self._recording_size = int(
            (simulation_duration_ms / error_window_size) *
            front_end_common_constants.BYTES_PER_WORD)

        # set up recording region IDs and data types
        self._region_ids = dict()
        self._region_dtypes = dict()
        for n in range(self._n_recordable_variables):
            self._region_ids[self.RECORDABLE_VARIABLES[n]] = n
            self._region_dtypes[
                self.RECORDABLE_VARIABLES[n]] = self.RECORDABLE_DTYPES[n]

        self._m_vertex = None

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

    def neurons(self):
        return self._n_neurons

    def get_maximum_delay_supported_in_ms(self, machine_time_step):
        # ICubVorEnv has no synapses so can simulate only one timestep of delay
        return machine_time_step / 1000.0

    # ------------------------------------------------------------------------
    # ApplicationVertex overrides
    # ------------------------------------------------------------------------
    @overrides(ApplicationVertex.get_resources_used_by_atoms)
    def get_resources_used_by_atoms(self, vertex_slice):
        # **HACK** only way to force no partitioning is to zero dtcm and cpu
        container = ResourceContainer(
            sdram=VariableSDRAM(fixed_sdram=0, per_timestep_sdram=4),
            dtcm=DTCMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0))

        return container

    @overrides(ApplicationVertex.create_machine_vertex)
    def create_machine_vertex(self, vertex_slice, resources_required,
                              label=None, constraints=None):
        # Return suitable machine vertex
        return ICubVorEnvMachineVertex(
            resources_required, constraints, self._label, self, vertex_slice)

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self):
        return self._n_neurons

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"machine_time_step": "MachineTimeStep",
                   "time_scale_factor": "TimeScaleFactor",
                   "routing_info": "MemoryRoutingInfos",
                   "tags": "MemoryTags"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"machine_time_step", "time_scale_factor",
                                     "routing_info", "tags"}
               )
    def generate_data_specification(self, spec, placement, machine_time_step,
                                    time_scale_factor, routing_info, tags):
        vertex = placement.vertex

        spec.comment("\n*** Spec for ICubVorEnv Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=ICubVorEnvMachineVertex._ICUB_VOR_ENV_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=ICubVorEnvMachineVertex._ICUB_VOR_ENV_REGIONS
                .ICUB_VOR_ENV.value,
            size=self.ICUB_VOR_ENV_REGION_BYTES, label='ICubVorEnvParams')
        # reserve recording region
        spec.reserve_memory_region(
            ICubVorEnvMachineVertex._ICUB_VOR_ENV_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(
                len(self.RECORDABLE_VARIABLES)))
        spec.reserve_memory_region(
            region=ICubVorEnvMachineVertex._ICUB_VOR_ENV_REGIONS.DATA.value,
            size=self.BASE_DATA_REGION_BYTES + (self._number_of_inputs * 16),
            label='ICubVorEnvArms')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            ICubVorEnvMachineVertex._ICUB_VOR_ENV_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # Write icub_vor_env region containing routing key to transmit with
        spec.comment("\nWriting icub_vor_env region:\n")
        spec.switch_write_focus(
            ICubVorEnvMachineVertex._ICUB_VOR_ENV_REGIONS.ICUB_VOR_ENV.value)
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.LIVE_POISSON_CONTROL_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting icub_vor_env recording region:\n")
        spec.switch_write_focus(
            ICubVorEnvMachineVertex._ICUB_VOR_ENV_REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        recording_sizes = [
            self._recording_size for _ in range(self._n_recordable_variables)]
        spec.write_array(recording_utilities.get_recording_header_array(
            recording_sizes, ip_tags=ip_tags))

        # Write parameters for ICubVorEnv data
        spec.comment("\nWriting icub_vor_env data region:\n")
        float_scale = float(DataType.S1615.scale)
        spec.switch_write_focus(
            ICubVorEnvMachineVertex._ICUB_VOR_ENV_REGIONS.DATA.value)
        spec.write_value(self._error_window_size, data_type=DataType.UINT32)
        spec.write_value(self._output_size, data_type=DataType.UINT32)
        spec.write_value(self._number_of_inputs, data_type=DataType.UINT32)
        spec.write_value(self.__round_to_nearest_accum(self._gain), data_type=DataType.S1615)
        spec.write_value(self.__round_to_nearest_accum(self._pos_to_vel), data_type=DataType.S1615)
        # Write the data - Arrays must be 32-bit values, so convert
        data = numpy.array(
            [int(x * float_scale) for x in self._perfect_eye_pos],
            dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))
        data = numpy.array(
            [int(x * float_scale) for x in self._perfect_eye_vel],
            dtype=numpy.uint32)
        spec.write_array(data.view(numpy.uint32))

        # End-of-Spec:
        spec.end_specification()

    def __round_to_nearest_accum(self, x):
        eps = 2. ** (-15)
        x_approx = numpy.floor((x / eps) + 0.5) * eps
        return x_approx
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
        for n in range(len(self.RECORDABLE_VARIABLES)):
            self._clear_recording_region(buffer_manager, placements, n)

    @overrides(AbstractNeuronRecordable.get_recordable_variables)
    def get_recordable_variables(self):
        return self.RECORDABLE_VARIABLES

    @overrides(AbstractNeuronRecordable.is_recording)
    def is_recording(self, variable):
        return True

    @overrides(AbstractNeuronRecordable.set_recording)
    def set_recording(self, variable, new_state=True, sampling_interval=None,
                      indexes=None):
        pass

    @overrides(AbstractNeuronRecordable.get_neuron_sampling_interval)
    def get_neuron_sampling_interval(self, variable):
        return 10000  # 10 seconds hard coded in as sim duration... ?

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(self, variable, n_machine_time_steps, placements,
                 buffer_manager, machine_time_step):
        if self._m_vertex is None:
            self._m_vertex = self.machine_vertices.pop()
        print('get_data from machine vertex ', self._m_vertex,
              ' for variable ', variable)
        placement = placements.get_placement_of_vertex(self._m_vertex)

        # Read the data recorded
        data_values, _ = buffer_manager.get_data_by_placement(
            placement, self._region_ids[variable])
        data = data_values

        numpy_format = list()
        output_format = list()
        if self._region_dtypes[variable] is DataType.S1615:
            numpy_format.append((variable, numpy.int32))
            output_format.append((variable, numpy.float32))
        else:
            numpy_format.append((variable, numpy.int32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)
        if self._region_dtypes[variable] is DataType.S1615:
            convert = numpy.zeros_like(
                output_data, dtype=numpy.float32).view(output_format)
            for i in range(output_data.size):
                for j in range(len(numpy_format)):
                    convert[i][j] = float(
                        output_data[i][j]) / float(DataType.S1615.scale)
            return convert
        else:
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
