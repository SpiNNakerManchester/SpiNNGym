# Copyright (c) 2019-2021 The University of Manchester
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

import numpy

from spinn_utilities.overrides import overrides
from spinn_utilities.config_holder import get_config_int

from data_specification.enums.data_type import DataType

# PACMAN imports
# from pacman.executor.injection_decorator import inject_items
from pacman.model.constraints.key_allocator_constraints import \
    ContiguousKeyRangeContraint
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)
from pacman.model.graphs.common import Slice
# from pacman.model.graphs.application import ApplicationVertex
# from pacman.model.resources.cpu_cycles_per_tick_resource import \
#     CPUCyclesPerTickResource
# from pacman.model.resources.dtcm_resource import DTCMResource
# from pacman.model.resources.resource_container import ResourceContainer
# from pacman.model.resources.variable_sdram import VariableSDRAM

# from data_specification.enums.data_type import DataType

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
# from spinn_front_end_common.utilities import globals_variables
# from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants
from spinn_front_end_common.utilities.exceptions import ConfigurationException

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import \
    AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
# from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

# ICubVorEnv imports
from spinn_gym.games.icub_vor_env.icub_vor_env_machine_vertex \
    import ICubVorEnvMachineVertex

NUMPY_DATA_ELEMENT_TYPE = numpy.double


# ----------------------------------------------------------------------------
# ICubVorEnv
# ----------------------------------------------------------------------------
class ICubVorEnv(AbstractOneAppOneMachineVertex,
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
    # error_window_size, output_size, number_of_inputs, gain, pos_to_vel,
    # wta_decision, low_error_rate and high_error_rate
    BASE_DATA_REGION_BYTES = 8 * 4
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
        # boosts the effect of individual spikes
        'gain': 20,
        # magic multiplier to convert movement delta to speed
        'pos_to_vel': 1 / (0.001 * 2 * numpy.pi * 10),
        'wta_decision': False,
        'low_error_rate': 2,  # Hz
        'high_error_rate': 20,  # Hz
        'output_size': 200,  # neurons encoding error via climbing fibres
        'constraints': None,
        'label': "ICubVorEnv",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION}

    def __init__(self, head_pos, head_vel, perfect_eye_pos, perfect_eye_vel,
                 error_window_size=default_parameters['error_window_size'],
                 output_size=default_parameters['output_size'],
                 gain=default_parameters['gain'],
                 pos_to_vel=default_parameters['pos_to_vel'],
                 wta_decision=default_parameters['wta_decision'],
                 low_error_rate=default_parameters['low_error_rate'],
                 high_error_rate=default_parameters['high_error_rate'],
                 constraints=default_parameters['constraints'],
                 label=default_parameters['label'],
                 incoming_spike_buffer_size=default_parameters[
                     'incoming_spike_buffer_size'],
                 simulation_duration_ms=default_parameters['duration']):
        """
        :param head_pos: array of head positions
        :param head_vel: array of head velocities
        :param perfect_eye_pos: array of ideal eye positions to produce VOR
        :param perfect_eye_vel: array of ideal eye velocities to produce VOR
        :param error_window_size: how often the environment changes
        :param output_size: numbers of neurons encoding the error transmitted \
            via combing fibres
        :param gain: boosts the effect of individual spikes
        :param pos_to_vel: magic multiplier to convert movement delta to speed
        :param wta_decision: whether eye movement takes into account the \
            difference in number of spikes between L and R
        :param constraints: usual sPyNNaker constraints
        :param label: name of the population
        :param incoming_spike_buffer_size:
        :param simulation_duration_ms: maximum simulation duration for this \
            application vertex
        """
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
        self._wta_decision = wta_decision
        self._low_error_rate = low_error_rate
        self._high_error_rate = high_error_rate
        self._number_of_inputs = len(perfect_eye_pos)
        if self._number_of_inputs != len(perfect_eye_vel):
            raise ConfigurationException(
                "The length of perfect_eye_pos {} is not the same as the "
                "length of perfect_eye_vel {}".format(
                    self._number_of_inputs, len(perfect_eye_vel)))

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

        resources_required = (
            self.ICUB_VOR_ENV_REGION_BYTES + self.BASE_DATA_REGION_BYTES +
            self._recording_size)

        vertex_slice = Slice(0, self._n_neurons - 1)

        # Superclasses
        super(ICubVorEnv, self).__init__(
            ICubVorEnvMachineVertex(
                vertex_slice, resources_required, constraints, label, self,
                head_pos, head_vel, perfect_eye_pos, perfect_eye_vel,
                error_window_size, output_size, gain, pos_to_vel, wta_decision,
                low_error_rate, high_error_rate, incoming_spike_buffer_size,
                simulation_duration_ms),
            label=label, constraints=constraints)

        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        SimplePopulationSettable.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractAcceptsIncomingSynapses.__init__(self)
        self._change_requires_mapping = True
        if incoming_spike_buffer_size is None:
            self._incoming_spike_buffer_size = get_config_int(
                "Simulation", "incoming_spike_buffer_size")

    def neurons(self):
        return self._n_neurons

    @property
    @overrides(AbstractOneAppOneMachineVertex.n_atoms)
    def n_atoms(self):
        return self._n_neurons

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
    def get_data(
            self, variable, n_machine_time_steps, placements, buffer_manager):
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

    def __str__(self):
        return "{} with {} atoms".format(self._label, self.n_atoms)

    def __repr__(self):
        return self.__str__()
