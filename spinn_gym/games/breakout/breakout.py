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

from __future__ import print_function
import numpy

from spinn_utilities.overrides import overrides

# PACMAN imports
from pacman.model.constraints.key_allocator_constraints import (
    ContiguousKeyRangeContraint)
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)
from pacman.model.graphs.common import Slice
from spinn_utilities.config_holder import get_config_int

# SpinnFrontEndCommon imports
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models. \
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.data import FecDataView
from spinn_front_end_common.utilities.utility_objs import ExecutableType

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import \
    AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

# Breakout imports
from spinn_gym.games.breakout.breakout_machine_vertex import \
    BreakoutMachineVertex


# ----------------------------------------------------------------------------
# Breakout
# ----------------------------------------------------------------------------
class Breakout(AbstractOneAppOneMachineVertex,
               AbstractProvidesOutgoingPartitionConstraints,
               AbstractAcceptsIncomingSynapses,
               AbstractNeuronRecordable,
               SimplePopulationSettable):

    @overrides(AbstractAcceptsIncomingSynapses.verify_splitter)
    def verify_splitter(self, splitter):
        # Need to ignore this verify
        pass

    @overrides(AbstractAcceptsIncomingSynapses.get_connections_from_machine)
    def get_connections_from_machine(
            self, app_edge, synapse_info):

        # TODO: make this work properly (the following call does nothing)

        super(Breakout, self).get_connections_from_machine(
            app_edge, synapse_info)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def clear_connection_cache(self):
        pass

    def get_synapse_id_by_target(self, target):
        return 0

    BREAKOUT_REGION_BYTES = 4
    PARAM_REGION_BYTES = 40
    WIDTH_PIXELS = 160
    X_FACTOR = 16
    HEIGHT_PIXELS = 128
    Y_FACTOR = 16
    COLOUR_BITS = 2
    MAX_SIM_DURATION = 1000*60*60*24*7  # 1 week in milliseconds
    rand_seed = [numpy.random.randint(10000),
                 numpy.random.randint(10000),
                 numpy.random.randint(10000),
                 numpy.random.randint(10000)]

    # parameters expected by PyNN
    default_parameters = {
        'x_factor': X_FACTOR,
        'y_factor': Y_FACTOR,
        'width': WIDTH_PIXELS,
        'height': HEIGHT_PIXELS,
        'colour_bits': COLOUR_BITS,
        'constraints': None,
        'label': "Breakout",
        'incoming_spike_buffer_size': None,
        'duration': MAX_SIM_DURATION,
        'bricking': 1,
        'random_seed': rand_seed
    }

    def __init__(self, x_factor=X_FACTOR, y_factor=Y_FACTOR,
                 width=WIDTH_PIXELS, height=HEIGHT_PIXELS,
                 colour_bits=COLOUR_BITS, constraints=None,
                 label="Breakout", incoming_spike_buffer_size=None,
                 simulation_duration_ms=MAX_SIM_DURATION, bricking=1,
                 random_seed=rand_seed):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._label = label
        self._x_factor = x_factor
        self._y_factor = y_factor
        self._width = width/x_factor
        self._height = height/y_factor
        self._colour_bits = colour_bits
        self._width_bits = numpy.uint32(numpy.ceil(numpy.log2(self._width)))
        self._height_bits = numpy.uint32(numpy.ceil(numpy.log2(self._height)))

        self._n_neurons = int(1 << (self._width_bits + self._height_bits +
                                    self._colour_bits))
        self._bricking = bricking
        self._rand_seed = random_seed

        # print self._rand_seed
        # print "# width =", self._width
        # print "# width bits =", self._width_bits
        # print "# height =", self._height
        # print "# height bits =", self._height_bits
        # print "# neurons =", self._n_neurons

        # Define size of recording region
        self._recording_size = int((simulation_duration_ms/10000.) * 4)

        # (static) resources required
        # technically as using OneAppOneMachine this is not necessary?
        resources_required = (
            self.BREAKOUT_REGION_BYTES + self.PARAM_REGION_BYTES +
            self._recording_size)

        vertex_slice = Slice(0, self._n_neurons - 1)

        # Superclasses
        super(Breakout, self).__init__(
            BreakoutMachineVertex(
                vertex_slice, resources_required, constraints, self._label,
                self, x_factor, y_factor, width, height, colour_bits,
                incoming_spike_buffer_size, simulation_duration_ms, bricking,
                random_seed),
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

        # **TODO** should we calculate this automatically
        # based on log2 of width and height?
        return self._n_neurons

    # ------------------------------------------------------------------------
    # AbstractHasAssociatedBinary overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        print("Breakout get_binary_file_name")
        return "breakout.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        print("Breakout get_binary_start_type")
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
            self, variable, buffer_manager):
        self._clear_recording_region(buffer_manager, 0)

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
    def get_data(self, variable, n_machine_time_steps, buffer_manager):
        vertex = self.machine_vertices.pop()
        placement = FecDataView().placements.get_placement_of_vertex(vertex)

        # Read the data recorded
        data_values, _ = buffer_manager.get_data_by_placement(placement, 0)
        data = data_values

        numpy_format = list()
        numpy_format.append(("Score", numpy.int32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

        return output_data

    def _clear_recording_region(self, buffer_manager, recording_region_id):
        """ Clear a recorded data region from the buffer manager.

        :param buffer_manager: the buffer manager object
        :param recording_region_id: the recorded region ID for clearing
        :rtype: None
        """
        placements = FecDataView().placements
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
