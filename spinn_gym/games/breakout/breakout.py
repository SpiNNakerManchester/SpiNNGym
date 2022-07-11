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

import numpy

from spinn_utilities.overrides import overrides

# common imports
from spinn_gym.games import SpinnGymApplicationVertex

# Breakout imports
from spinn_gym.games.breakout.breakout_machine_vertex import \
    BreakoutMachineVertex


# ----------------------------------------------------------------------------
# Breakout
# ----------------------------------------------------------------------------
class Breakout(SpinnGymApplicationVertex):

    ONE_WEEK_IN_MS = 1000*60*60*24*7
    RANDOM_SEED = [numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000)]

    __slots__ = []

    def __init__(self, x_factor=16, y_factor=16, width=160, height=128,
                 colour_bits=2, constraints=None,  label="Breakout",
                 simulation_duration_ms=ONE_WEEK_IN_MS, bricking=1,
                 random_seed=None):
        if random_seed is None:
            random_seed = list(self.RANDOM_SEED)

        width_bits = numpy.uint32(numpy.ceil(numpy.log2(width/x_factor)))
        height_bits = numpy.uint32(numpy.ceil(numpy.log2(height/y_factor)))

        n_neurons = int(1 << (width_bits + height_bits + colour_bits))

        machne_vertex = BreakoutMachineVertex(
            label, constraints, self, n_neurons, simulation_duration_ms,
            random_seed, x_factor, y_factor, colour_bits, bricking)

        # Superclasses
        super(Breakout, self).__init__(
            machne_vertex,  label, constraints, n_neurons)

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self):
        return numpy.int32
