from enum import Enum

from spinn_utilities.overrides import overrides

# PACMAN imports
from pacman.model.graphs.machine import MachineVertex

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.interface.buffer_management.buffer_models import (
    AbstractReceiveBuffersToHost)
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.utilities.utility_objs import ExecutableType


# ----------------------------------------------------------------------------
# BanditMachineVertex
# ----------------------------------------------------------------------------
class BanditMachineVertex(MachineVertex, AbstractReceiveBuffersToHost,
                          AbstractHasAssociatedBinary):
    _BANDIT_REGIONS = Enum(
        value="_BANDIT_REGIONS",
        names=[('SYSTEM', 0),
               ('BANDIT', 1),
               ('RECORDING', 2),
               ('ARMS', 3)])

    def __init__(self, resources_required, constraints=None, label=None,
                 app_vertex=None, vertex_slice=None):
        # Superclasses
        MachineVertex.__init__(
            self, label, constraints, app_vertex, vertex_slice)
        self._resource_required = resources_required

    @property
    def resources_required(self):
        return self._resource_required

    def get_minimum_buffer_sdram_usage(self):
        return 0  # probably should make this a user input

    def get_recorded_region_ids(self):
        return [0]

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._BANDIT_REGIONS.RECORDING.value, txrx)

    def get_n_keys_for_partition(self, partition):
        return 8  # eight for control IDs

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "bandit.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        # return ExecutableStartType.USES_SIMULATION_INTERFACE
        return ExecutableType.USES_SIMULATION_INTERFACE
