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

# PACMAN imports
from pacman.model.graphs.common import Slice

# sPyNNaker imports
from spynnaker.pyNN.models.common import AbstractNeuronRecordable

# common imports
from spinn_gym.games import SpinnGymApplicationVertex

# Breakout imports
from spinn_gym.games.breakout.breakout_machine_vertex import \
    BreakoutMachineVertex


# ----------------------------------------------------------------------------
# Breakout
# ----------------------------------------------------------------------------
class Breakout(SpinnGymApplicationVertex):

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

    # ------------------------------------------------------------------------
    # Recording overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractNeuronRecordable.get_data)
    def get_data(
            self, variable, n_machine_time_steps, placements, buffer_manager):
        vertex = self.machine_vertices.pop()
        placement = placements.get_placement_of_vertex(vertex)

        # Read the data recorded
        data_values, _ = buffer_manager.get_data_by_placement(placement, 0)
        data = data_values

        numpy_format = list()
        numpy_format.append(("Score", numpy.int32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

        return output_data
