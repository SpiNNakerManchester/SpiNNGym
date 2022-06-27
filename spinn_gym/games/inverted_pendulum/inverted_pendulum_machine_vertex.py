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

from enum import Enum

from spinn_utilities.overrides import overrides

from data_specification.enums.data_type import DataType

# PACMAN imports
from pacman.executor.injection_decorator import inject_items

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.abstract_models \
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants

# sPyNNaker imports
from spynnaker.pyNN.utilities import constants

# spinn_gym imports
from spinn_gym.games import SpinnGymMachineVertex


# ----------------------------------------------------------------------------
# PendulumMachineVertex
# ----------------------------------------------------------------------------
class PendulumMachineVertex(SpinnGymMachineVertex):
    PENDULUM_REGION_BYTES = 4
    DATA_REGION_BYTES = 15 * 4

    _PENDULUM_REGIONS = Enum(
        value="_PENDULUM_REGIONS",
        names=[('SYSTEM', 0),
               ('PENDULUM', 1),
               ('RECORDING', 2),
               ('DATA', 3)])

    __slots__ = ["_bin_overlap", "_central", "_encoding", "_force_increments",
                 "_max_firing_rate", "_number_of_bins", "_pole_angle",
                 "_pole_length", "_reward_based", "_tau_force",
                 "_time_increment"]

    def __init__(self, label, constraints, app_vertex, n_neurons,
                 simulation_duration_ms, random_seed,
                 encoding, time_increment, pole_length, pole_angle,
                 reward_based, force_increments, max_firing_rate,
                 number_of_bins, central, bin_overlap, tau_force):
        """

        :param label: The optional name of the vertex
        :type label: str or None
        :param iterable(AbstractConstraint) constraints:
            The optional initial constraints of the vertex
        :type constraints: iterable(AbstractConstraint) or None
        :type constraints: iterable(AbstractConstraint)  or None
        :param app_vertex:
            The application vertex that caused this machine vertex to be
            created. If None, there is no such application vertex.
        :type app_vertex: ApplicationVertex or None
        :param int n_neurons:
            The number of neurons to be used to create the slice of the
            application vertex that this machine vertex implements.
        :param int region_bytes: The bytes needed other than recording
        :param float simulation_duration_ms:
        :param list(int) random_seed: List of 4 vlaues to seed the c code
        :param encoding:
        :param time_increment:
        :param pole_length:
        :param pole_angle:
        :param reward_based:
        :param force_increments:
        :param max_firing_rate:
        :param number_of_bins:
        :param central:
        :param bin_overlap:
        :param tau_force:

        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        :raises PacmanValueError: If the slice of the machine_vertex is too big
        :raise AttributeError:
            If a not None app_vertex is not an ApplicationVertex
        """

        # Superclasses
        super(PendulumMachineVertex, self).__init__(
            label, constraints, app_vertex, n_neurons,
            self.PENDULUM_REGION_BYTES + self.DATA_REGION_BYTES,
            simulation_duration_ms,  random_seed)

        self._encoding = encoding

        # Pass in variables
        self._pole_length = pole_length
        self._pole_angle = pole_angle

        self._force_increments = force_increments

        self._time_increment = time_increment
        self._reward_based = reward_based

        self._max_firing_rate = max_firing_rate
        self._number_of_bins = number_of_bins
        self._central = central
        self._bin_overlap = bin_overlap
        self._tau_force = tau_force

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"routing_info": "RoutingInfos"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"routing_info"}
               )
    def generate_data_specification(self, spec, placement, routing_info):
        # pylint: disable=arguments-differ
        vertex = placement.vertex

        spec.comment("\n*** Spec for Pendulum Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=self._PENDULUM_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=self._PENDULUM_REGIONS.PENDULUM.value,
            size=self.PENDULUM_REGION_BYTES, label='PendulumVertex')
        # reserve recording region
        spec.reserve_memory_region(
            self._PENDULUM_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=self._PENDULUM_REGIONS.DATA.value,
            size=self.DATA_REGION_BYTES, label='PendulumData')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            self._PENDULUM_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_bin# Copyright (c) 2019-2022 The University of Manchester
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

# PACMAN imports
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)
from pacman.model.graphs.common import Slice
from spinn_utilities.config_holder import get_config_int

# SpinnFrontEndCommon imports
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import \
    AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

# Pendulum imports
from spinn_gym.games.double_inverted_pendulum.double_pendulum_machine_vertex \
    import DoublePendulumMachineVertex


# ----------------------------------------------------------------------------
# Double Pendulum
# ----------------------------------------------------------------------------
class DoublePendulum(AbstractOneAppOneMachineVertex,
                     AbstractAcceptsIncomingSynapses, AbstractNeuronRecordable,
                     SimplePopulationSettable):

    __slots__ = []

    def __init__(
            self, constraints=None, encoding=0, time_increment=20,
            pole_length=1.0, pole_angle=0.1, pole2_length=0, pole2_angle=0,
            reward_based=1, force_increments=100, max_firing_rate=100,
            number_of_bins=20, central=1, random_seed=None, bin_overlap=2,
            tau_force=0, label="pole", simulation_duration_ms=ONE_WEEK_IN_MS):
        """

        :param constraints:
        :param encoding:  0 rate, 1 receptive bins, 2 spike time, 3 rank
        :param time_increment:
        :param pole_length:
        :param pole_angle:
        :param pole2_length:
        :param pole2_angle:
        :param reward_based:
        :param force_increments:
        :param max_firing_rate:
        :param number_of_bins:
        :param central:
        :param random_seed:
        :param bin_overlap:
        :param tau_force:
        :param label:
        :param simulation_duration_ms:
        """

        if random_seed is None:
            random_seed = list(self.RANDOM_SEED)

        # for rate based it's only 1 neuron per metric
        # (position, angle, velocity of both)
        n_neurons = 6 * number_of_bins

        machine_vertex = DoublePendulumMachineVertex(
            label, constraints, self, n_neurons, simulation_duration_ms,
            random_seed, encoding, time_increment, pole_length, pole_angle,
            pole2_length, pole2_angle, reward_based, force_increments,
            max_firing_rate, number_of_bins, central, bin_overlap, tau_force)

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

    @overrides(AbstractNeuronRecordable.get_neuron_sampling_interval)
    def get_neuron_sampling_interval(self, variable):
        return 10000  # 10 seconds hard coded in bkout.c

    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(
            self, variable, n_machine_time_steps, placements, buffer_manager):
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
ary_file_name()))

        # Write pendulum region containing routing key to transmit with
        spec.comment("\nWriting pendulum region:\n")
        spec.switch_write_focus(
            self._PENDULUM_REGIONS.PENDULUM.value)
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.SPIKE_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting pendulum recording region:\n")
        spec.switch_write_focus(
            self._PENDULUM_REGIONS.RECORDING.value)
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size]))

        # Write pendulum data
        spec.comment("\nWriting pendulum data region:\n")
        spec.switch_write_focus(
            self._PENDULUM_REGIONS.DATA.value)
        spec.write_value(self._encoding, data_type=DataType.UINT32)
        spec.write_value(self._time_increment, data_type=DataType.UINT32)
        spec.write_value(self._pole_length, data_type=DataType.S1615)
        spec.write_value(self._pole_angle, data_type=DataType.S1615)
        spec.write_value(self._reward_based, data_type=DataType.UINT32)
        spec.write_value(self._force_increments, data_type=DataType.UINT32)
        spec.write_value(self._max_firing_rate, data_type=DataType.UINT32)
        spec.write_value(self._number_of_bins, data_type=DataType.UINT32)
        spec.write_value(self._central, data_type=DataType.UINT32)
        spec.write_value(self._random_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._random_seed[3], data_type=DataType.UINT32)
        spec.write_value(self._bin_overlap, data_type=DataType.S1615)
        spec.write_value(self._tau_force, data_type=DataType.S1615)

        # End-of-Spec:
        spec.end_specification()

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._PENDULUM_REGIONS.RECORDING.value, txrx)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "inverted_pendulum.aplx"
