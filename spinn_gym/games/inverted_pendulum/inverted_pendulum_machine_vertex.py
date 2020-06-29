from enum import Enum

# SpinnFrontEndCommon imports
from pacman.model.graphs.machine import MachineVertex
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.interface.buffer_management.buffer_models import (
    AbstractReceiveBuffersToHost)


# ----------------------------------------------------------------------------
# PendulumMachineVertex
# ----------------------------------------------------------------------------
class PendulumMachineVertex(MachineVertex, AbstractReceiveBuffersToHost):
    _PENDULUM_REGIONS = Enum(
        value="_PENDULUM_REGIONS",
        names=[('SYSTEM', 0),
               ('PENDULUM', 1),
               ('RECORDING', 2),
               ('DATA', 3)])

    def __init__(self, resources_required, constraints=None, label=None):
        # Superclasses
        MachineVertex.__init__(self, label,
                               constraints=constraints)
        self._resource_required = resources_required

    @property
    def resources_required(self):
        return self._resource_required

    def get_minimum_buffer_sdram_usage(self):
        return 0  # probably should make this a user input

    def get_n_timesteps_in_buffer_space(self, buffer_space, machine_time_step):
        return recording_utilities.get_n_timesteps_in_buffer_space(
            buffer_space, [0])  # this could be completely wrong - test

    def get_recorded_region_ids(self):
        return [0]

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._PENDULUM_REGIONS.RECORDING.value, txrx)
