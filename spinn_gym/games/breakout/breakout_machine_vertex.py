from enum import Enum

from spinn_utilities.overrides import overrides

# PACMAN imports
from pacman.model.graphs.machine import MachineVertex

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.interface.buffer_management.buffer_models.\
    abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary

# ----------------------------------------------------------------------------
# BreakoutMachineVertex
# ----------------------------------------------------------------------------
class BreakoutMachineVertex(MachineVertex, AbstractReceiveBuffersToHost,
                            AbstractHasAssociatedBinary):
    _BREAKOUT_REGIONS = Enum(
        value="_BREAKOUT_REGIONS",
        names=[('SYSTEM', 0),
               ('BREAKOUT', 1),
               ('RECORDING', 2),
               ('PARAMS', 3)])

    def __init__(self, resources_required, constraints=None, label=None,
                 app_vertex=None, vertex_slice=None):
        # Superclasses
        MachineVertex.__init__(
            self, label, constraints, app_vertex, vertex_slice)
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

    # ------------------------------------------------------------------------
    # AbstractHasAssociatedBinary overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        print("BreakoutMachineVertex get_binary_file_name")
        return "breakout.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        print("BreakoutMachineVertex get_binary_start_type")
        return ExecutableType.USES_SIMULATION_INTERFACE

