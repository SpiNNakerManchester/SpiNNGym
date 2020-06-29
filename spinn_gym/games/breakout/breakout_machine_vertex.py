from enum import Enum

# PACMAN imports
from pacman.model.graphs.machine import MachineVertex

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.interface.buffer_management.buffer_models.\
    abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost


# ----------------------------------------------------------------------------
# BreakoutMachineVertex
# ----------------------------------------------------------------------------
class BreakoutMachineVertex(MachineVertex, AbstractReceiveBuffersToHost):
    _BREAKOUT_REGIONS = Enum(
        value="_BREAKOUT_REGIONS",
        names=[('SYSTEM', 0),
               ('BREAKOUT', 1),
               ('RECORDING', 2),
               ('PARAMS', 3)])

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

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._BREAKOUT_REGIONS.RECORDING.value, txrx)

    def get_recorded_region_ids(self):
        """ Get the recording region IDs that have been recorded using buffering

        :return: The region numbers that have active recording
        :rtype: iterable(int) """
        return [0]
