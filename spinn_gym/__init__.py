# from model_binaries import __file__ as binaries_path
from games.breakout.breakout import Breakout
from games.multi_arm_bandit.bandit import Bandit
# from visualiser.visualiser import Visualiser
# from visualiser.visualiser_subsamp import Visualiser_subsamp

from spynnaker.pyNN.abstract_spinnaker_common import AbstractSpiNNakerCommon

# import __file__
# binary_path = os.path.abspath(__file__) +
import os

binary_path = os.path.join(os.path.split(__file__)[0], 'model_binaries')
print binary_path
# print os.path.join(__file__, os.path.pardir+'/model_binaries')

# AbstractSpiNNakerCommon.register_binary_search_path(os.path.dirname(binaries_path))
AbstractSpiNNakerCommon.register_binary_search_path(binary_path)

# This adds the model binaries path to the paths searched by sPyNNaker
# ef = executable_finder.ExecutableFinder(os.path.dirname(binaries_path))
# executable_finder.add_path(os.path.dirname(binaries_path))