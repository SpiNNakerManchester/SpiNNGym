# Copyright (c) 2019 The University of Manchester
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy

from spinn_utilities.abstract_base import abstractproperty
from spinn_utilities.overrides import overrides

# PACMAN imports
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)

# sPyNNaker imports
from spynnaker.pyNN.models.common import PopulationApplicationVertex
from spynnaker.pyNN.data import SpynnakerDataView


class SpinnGymApplicationVertex(
        AbstractOneAppOneMachineVertex,
        PopulationApplicationVertex):

    __slots__ = ()

    def __init__(self, machine_vertex, label, n_atoms):
        """
        Creates an ApplicationVertex which has exactly one predefined \
        MachineVertex

        :param machine_vertex: MachineVertex
        :param str label: The optional name of the vertex.
        :type constraints: iterable(AbstractConstraint) or None
        :raise PacmanInvalidParameterException:
            If one of the constraints is not valid
        """
        super(SpinnGymApplicationVertex, self).__init__(
            machine_vertex, label, n_atoms)

    @overrides(PopulationApplicationVertex.get_units)
    def get_units(self, name: str) -> str:
        if name == "score":
            return ""
        return super(SpinnGymApplicationVertex, self).get_units(name)

    def get_recorded_data(self, name):
        if name != "score":
            raise KeyError(f"{name} was not recorded")

        vertex = self.machine_vertices.pop()
        placement = SpynnakerDataView.get_placement_of_vertex(vertex)
        buffer_manager = SpynnakerDataView.get_buffer_manager()

        # Read the data recorded
        data_values, _ = buffer_manager.get_data_by_placement(placement, 0)
        data = data_values

        numpy_format = list()
        numpy_format.append(("Score", self.score_format))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)

        # return formatted_data
        return output_data

    def describe(self):
        """ Get a human-readable description of the cell or synapse type.

        The output may be customised by specifying a different template
        together with an associated template engine
        (see :py:mod:`pyNN.descriptions`).

        If template is None, then a dictionary containing the template context
        will be returned.

        :rtype: dict(str, ...)
        """

        context = {
            "name": self.__class__.__name__
        }
        return context

    @abstractproperty
    def score_format(self) -> type:
        """
        The numpy format for the scores data
        """

    def __str__(self):
        return "{} with {} atoms".format(self._label, self.n_atoms)

    def __repr__(self):
        return self.__str__()
