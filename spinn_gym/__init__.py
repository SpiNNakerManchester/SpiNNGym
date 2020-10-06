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
