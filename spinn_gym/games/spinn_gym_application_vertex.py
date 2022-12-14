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

from spinn_utilities.abstract_base import abstractproperty
from spinn_utilities.overrides import overrides

# PACMAN imports
from pacman.model.graphs.application.abstract import (
    AbstractOneAppOneMachineVertex)

# sPyNNaker imports
from spynnaker.pyNN.models.common import (
    PopulationApplicationVertex, RecordingType)
from spynnaker.pyNN.data import SpynnakerDataView


class SpinnGymApplicationVertex(
        AbstractOneAppOneMachineVertex,
        PopulationApplicationVertex):

    __slots__ = []

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
    def get_units(self, name):
        if name == "score":
            return ""
        return super(SpinnGymApplicationVertex, self).get_units(name)

    @overrides(PopulationApplicationVertex.get_recordable_variables)
    def get_recordable_variables(self):
        return ["score"]

    @overrides(PopulationApplicationVertex.can_record)
    def can_record(self, name):
        return name == "score"

    @overrides(PopulationApplicationVertex.set_recording)
    def set_recording(self, name, sampling_interval=None, indices=None):
        if name != "score":
            raise KeyError(f"Cannot record {name}")

        if sampling_interval is not None:
            raise KeyError(
                "Sampling interval is not supported (fixed at 10000)")

        if indices is not None:
            raise KeyError("Indices are not supported")

        # No need to do anything, as always recording anyway!

    @overrides(PopulationApplicationVertex.set_not_recording)
    def set_not_recording(self, name, indices=None):
        if name != "score":
            raise KeyError(f"Cannot record {name}")

        if indices is not None:
            raise KeyError("Indices are not supported")

        # No need to do anything, as always recording anyway!

    @overrides(PopulationApplicationVertex.get_recording_variables)
    def get_recording_variables(self):
        return ["score"]

    @overrides(PopulationApplicationVertex.is_recording_variable)
    def is_recording_variable(self, name):
        return name == "score"

    @overrides(PopulationApplicationVertex.get_recorded_data)
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

    @overrides(PopulationApplicationVertex.get_recording_sampling_interval)
    def get_recording_sampling_interval(self, name):
        if name != "score":
            raise KeyError(f"Cannot record {name}")
        # recording is done at 10000ms intervals
        return 10000

    @overrides(PopulationApplicationVertex.get_recording_indices)
    def get_recording_indices(self, name):
        # Only the score is recorded
        return [0]

    @overrides(PopulationApplicationVertex.get_recording_type)
    def get_recording_type(self, name):
        if name != "score":
            raise KeyError(f"Cannot record {name}")
        return RecordingType.MATRIX

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
    def score_format(self):
        """
        The numpy format for the scores data
        """

    def __str__(self):
        return "{} with {} atoms".format(self._label, self.n_atoms)

    def __repr__(self):
        return self.__str__()
