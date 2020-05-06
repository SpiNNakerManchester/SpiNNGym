from enum import Enum

# PACMAN imports
# from pacman.model.decorators.overrides import overrides
from spinn_utilities.overrides import overrides


# SpinnFrontEndCommon imports
from pacman.model.graphs.machine import MachineVertex
from spinn_front_end_common.interface.provenance \
    .provides_provenance_data_from_machine_impl \
    import ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.interface.buffer_management.buffer_models import (
    AbstractReceiveBuffersToHost)

# ----------------------------------------------------------------------------
# LogicMachineVertex
# ----------------------------------------------------------------------------
class LogicMachineVertex(MachineVertex, AbstractReceiveBuffersToHost):
    _LOGIC_REGIONS = Enum(
        value="_LOGIC_REGIONS",
        names=[('SYSTEM', 0),
               ('LOGIC', 1),
               ('RECORDING', 2),
               ('DATA', 3)])

    def __init__(self, resources_required, constraints=None, label=None):
        # Superclasses
        MachineVertex.__init__(self, label,
                               constraints=constraints)
        # ProvidesProvenanceDataFromMachineImpl.__init__(
        #     self, self._BREAKOUT_REGIONS.PROVENANCE.value, 0)
        self._resource_required = resources_required

    @property
    def resources_required(self):
        return self._resource_required

    def get_minimum_buffer_sdram_usage(self):
        return 0  # probably should make this a user input

    def get_n_timesteps_in_buffer_space(self, buffer_space, machine_time_step):
        return recording_utilities.get_n_timesteps_in_buffer_space(
            buffer_space, [0])  # this could be completely wrong - test the value used

    def get_recorded_region_ids(self):
        return [0]

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._LOGIC_REGIONS.RECORDING.value, txrx)