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

from spinn_gym.games.breakout.breakout import Breakout
from spinn_gym.games.multi_arm_bandit.bandit import Bandit
from spinn_gym.games.inverted_pendulum.inverted_pendulum import Pendulum
from spinn_gym.games.logic.logic import Logic
from spinn_gym.games.store_recall.store_recall import Recall
from spinn_gym.games.double_inverted_pendulum.double_pendulum \
    import DoublePendulum
import os

# Put model_binaries directory on path
from spynnaker.pyNN.abstract_spinnaker_common import AbstractSpiNNakerCommon
binary_path = os.path.join(os.path.split(__file__)[0], 'model_binaries')
AbstractSpiNNakerCommon.register_binary_search_path(binary_path)

__all__ = ['Breakout', 'Bandit', 'Pendulum', 'Logic', 'Recall',
           'DoublePendulum']
