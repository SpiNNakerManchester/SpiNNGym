from games.breakout.breakout import Breakout
from games.multi_arm_bandit.bandit import Bandit
#from games.inverted_pendulum.inverted_pendulum import Pendulum
#from games.logic.logic import Logic
from games.store_recall.store_recall import Recall
#from games.double_inverted_pendulum.double_pendulum import DoublePendulum
import os

# Put model_binaries directory on path
from spynnaker.pyNN.abstract_spinnaker_common import AbstractSpiNNakerCommon
binary_path = os.path.join(os.path.split(__file__)[0], 'model_binaries')
AbstractSpiNNakerCommon.register_binary_search_path(binary_path)
