from enum import Enum

# PACMAN imports
from pacman.model.decorators.overrides import overrides


# SpinnFrontEndCommon imports
from pacman.model.graphs.machine import MachineVertex
from spinn_front_end_common.interface.provenance \
    .provides_provenance_data_from_machine_impl \
    import ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.utilities import helpful_functions, constants
from spinn_front_end_common.interface.buffer_management.buffer_models.abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost


# ----------------------------------------------------------------------------
# BanditMachineVertex
# ----------------------------------------------------------------------------
class DoublePendulumMachineVertex(MachineVertex, AbstractReceiveBuffersToHost):
    _DOUBLE_PENDULUM_REGIONS = Enum(
        value="_DOUBLE_PENDULUM_REGIONS",
        names=[('SYSTEM', 0),
               ('PENDULUM', 1),
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

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._DOUBLE_PENDULUM_REGIONS.RECORDING.value, txrx)

    def get_recorded_region_ids(self):
        """ Get the recording region IDs that have been recorded using buffering

        :return: The region numbers that have active recording
        :rtype: iterable(int) """
        return [0]