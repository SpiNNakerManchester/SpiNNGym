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

from spinn_front_end_common.interface.ds import DataType
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants
from spinn_front_end_common.utilities.exceptions import ConfigurationException

from spynnaker.pyNN.data import SpynnakerDataView

from spinn_gym.games import SpinnGymApplicationVertex

# ICubVorEnv imports
from spinn_gym.games.icub_vor_env.icub_vor_env_machine_vertex \
    import ICubVorEnvMachineVertex


# ----------------------------------------------------------------------------
# ICubVorEnv
# ----------------------------------------------------------------------------
class ICubVorEnv(SpinnGymApplicationVertex):

    # not sure this is entirely necessary but keeping it for now
    MAX_SIM_DURATION = 10000
    RANDOM_SEED = [numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000),
                   numpy.random.randint(10000)]
    # Probably better ways of doing this too, but keeping it for now
    RECORDABLE_VARIABLES = [
        "l_count", "r_count", "error", "eye_pos", "eye_vel"]
    RECORDABLE_DTYPES = [
        DataType.UINT32, DataType.UINT32, DataType.S1615, DataType.S1615,
        DataType.S1615]

    # magic multiplier to convert movement delta to speed
    POS_TO_VEL = 1 / (0.001 * 2 * numpy.pi * 10)

    def __init__(self, head_pos, head_vel, perfect_eye_pos, perfect_eye_vel,
                 error_window_size=10, output_size=200, gain=20,
                 pos_to_vel=POS_TO_VEL, wta_decision=False, low_error_rate=2,
                 high_error_rate=20, label="ICubVorEnv",
                 simulation_duration_ms=MAX_SIM_DURATION, random_seed=None):
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
        :param simulation_duration_ms: maximum simulation duration for this \
            application vertex
        """
        self._number_of_inputs = len(perfect_eye_pos)
        if self._number_of_inputs != len(perfect_eye_vel):
            raise ConfigurationException(
                "The length of perfect_eye_pos {} is not the same as the "
                "length of perfect_eye_vel {}".format(
                    self._number_of_inputs, len(perfect_eye_vel)))

        if random_seed is None:
            random_seed = list(self.RANDOM_SEED)

        # n_neurons is the number of atoms in the network, which in this
        # case only needs to be 2 (for receiving "left" and "right")
        n_neurons = 2

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
        machine_vertex = ICubVorEnvMachineVertex(
                label, self, n_neurons, simulation_duration_ms,
                random_seed, head_pos, head_vel, perfect_eye_pos,
                perfect_eye_vel, error_window_size, output_size, gain,
                pos_to_vel, wta_decision, low_error_rate, high_error_rate)

        super(ICubVorEnv, self).__init__(
            machine_vertex, label, n_neurons)

    # ------------------------------------------------------------------------
    # Recording overrides
    # ------------------------------------------------------------------------
    # @overrides(SpinnGymApplicationVertex.clear_recording_data)
    # def clear_recording_data(self, name):
    #     if name not in self.RECORDABLE_VARIABLES:
    #         raise KeyError(f"Cannot clear recording for {name}")
    #     for machine_vertex in self.machine_vertices:
    #         placement = SpynnakerDataView.get_placement_of_vertex(
    #             machine_vertex)
    #         buffer_manager = SpynnakerDataView.get_buffer_manager()
    #         buffer_manager.clear_recorded_data(
    #             placement.x, placement.y, placement.p,
    #             self.RECORDABLE_VARIABLES.index(name))
    #
    # @overrides(SpinnGymApplicationVertex.get_recordable_variables)
    # def get_recordable_variables(self):
    #     return self.RECORDABLE_VARIABLES

    @overrides(SpinnGymApplicationVertex.get_recorded_data)
    def get_recorded_data(self, name):
        if self._m_vertex is None:
            self._m_vertex = self.machine_vertices.pop()
        print('get_data from machine vertex ', self._m_vertex,
              ' for variable ', name)
        placement = SpynnakerDataView.get_placement_of_vertex(self._m_vertex)
        buffer_manager = SpynnakerDataView.get_buffer_manager()

        # Read the data recorded
        data_values, _ = buffer_manager.get_data_by_placement(
            placement, self._region_ids[name])
        data = data_values

        numpy_format = list()
        output_format = list()
        if self._region_dtypes[name] is DataType.S1615:
            numpy_format.append((name, numpy.int32))
            output_format.append((name, numpy.float32))
        else:
            numpy_format.append((name, numpy.int32))

        output_data = numpy.array(data, dtype=numpy.uint8).view(numpy_format)
        if self._region_dtypes[name] is DataType.S1615:
            convert = numpy.zeros_like(
                output_data, dtype=numpy.float32).view(output_format)
            for i in range(output_data.size):
                for j in range(len(numpy_format)):
                    convert[i][j] = float(
                        output_data[i][j]) / float(DataType.S1615.scale)
            return convert
        else:
            return output_data

    @property
    @overrides(SpinnGymApplicationVertex.score_format)
    def score_format(self):
        return numpy.int32
