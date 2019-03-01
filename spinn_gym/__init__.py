# from model_binaries import __file__ as binaries_path
from games.breakout.breakout import Breakout
# from visualiser.visualiser import Visualiser
# from visualiser.visualiser_subsamp import Visualiser_subsamp

from spynnaker.pyNN.abstract_spinnaker_common import AbstractSpiNNakerCommon

import os
binary_path=os.path.join(os.getcwd(), 'model_binaries')

# AbstractSpiNNakerCommon.register_binary_search_path(os.path.dirname(binaries_path))
AbstractSpiNNakerCommon.register_binary_search_path(binary_path)

# This adds the model binaries path to the paths searched by sPyNNaker
# ef = executable_finder.ExecutableFinder(os.path.dirname(binaries_path))
# executable_finder.add_path(os.path.dirname(binaries_path))