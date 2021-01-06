import numpy
from enum import Enum

from spinn_utilities.overrides import overrides

from data_specification.enums.data_type import DataType

# PACMAN imports
from pacman.executor.injection_decorator import inject_items
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ConstantSDRAM, ResourceContainer

# SpinnFrontEndCommon imports
from spinn_front_end_common.utilities import helpful_functions
from spinn_front_end_common.interface.buffer_management.buffer_models.\
    abstract_receive_buffers_to_host import AbstractReceiveBuffersToHost
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.abstract_models.\
    abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.interface.buffer_management \
    import recording_utilities
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants

from spynnaker.pyNN.utilities import constants


# ----------------------------------------------------------------------------
# BreakoutMachineVertex
# ----------------------------------------------------------------------------
class BreakoutMachineVertex(MachineVertex, AbstractGeneratesDataSpecification,
                            AbstractReceiveBuffersToHost,
                            AbstractHasAssociatedBinary):
    BREAKOUT_REGION_BYTES = 4
    PARAM_REGION_BYTES = 40

    _BREAKOUT_REGIONS = Enum(
        value="_BREAKOUT_REGIONS",
        names=[('SYSTEM', 0),
               ('BREAKOUT', 1),
               ('RECORDING', 2),
               ('PARAMS', 3)])

    def __init__(self, vertex_slice, resources_required, constraints, label,
                 app_vertex, x_factor, y_factor, width, height, colour_bits,
                 incoming_spike_buffer_size, simulation_duration_ms, bricking,
                 random_seed):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._resources_required = ResourceContainer(
            sdram=ConstantSDRAM(resources_required))

        self._label = label
        self._x_factor = x_factor
        self._y_factor = y_factor
        self._width = width/x_factor
        self._height = height/y_factor
        self._colour_bits = colour_bits
        self._width_bits = numpy.uint32(numpy.ceil(numpy.log2(self._width)))
        self._height_bits = numpy.uint32(numpy.ceil(numpy.log2(self._height)))

        self._n_neurons = (1 << (self._width_bits + self._height_bits +
                                 self._colour_bits))
        self._bricking = bricking
        self._rand_seed = random_seed

        # print self._rand_seed
        # print "# width =", self._width
        # print "# width bits =", self._width_bits
        # print "# height =", self._height
        # print "# height bits =", self._height_bits
        # print "# neurons =", self._n_neurons

        # Define size of recording region
        self._recording_size = int((simulation_duration_ms/10000.) * 4)

        # Superclasses
        MachineVertex.__init__(
            self, label, constraints, app_vertex, vertex_slice)
#         self._resource_required = resources_required

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"machine_time_step": "MachineTimeStep",
                   "time_scale_factor": "TimeScaleFactor",
                   "routing_info": "MemoryRoutingInfos",
                   "tags": "MemoryTags",
                   "n_machine_time_steps": "DataNTimeSteps"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"machine_time_step", "time_scale_factor",
                                     "routing_info", "tags",
                                     "n_machine_time_steps"}
               )
    def generate_data_specification(self, spec, placement, machine_time_step,
                                    time_scale_factor,
                                    routing_info, tags, n_machine_time_steps):
        vertex = placement.vertex

        spec.comment("\n*** Spec for Breakout Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=BreakoutMachineVertex._BREAKOUT_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')
        spec.reserve_memory_region(
            region=BreakoutMachineVertex._BREAKOUT_REGIONS.BREAKOUT.value,
            size=self.BREAKOUT_REGION_BYTES, label='BreakoutKey')
        # Reserve recording region
        spec.reserve_memory_region(
            BreakoutMachineVertex._BREAKOUT_REGIONS.RECORDING.value,
            recording_utilities.get_recording_header_size(1))
        spec.reserve_memory_region(
            region=BreakoutMachineVertex._BREAKOUT_REGIONS.PARAMS.value,
            size=self.PARAM_REGION_BYTES, label='Parameters')

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            BreakoutMachineVertex._BREAKOUT_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            vertex.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # Write breakout region containing routing key to transmit with
        spec.comment("\nWriting breakout region:\n")
        spec.switch_write_focus(
            BreakoutMachineVertex._BREAKOUT_REGIONS.BREAKOUT.value)
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.SPIKE_PARTITION_ID))

        # Write recording region for score
        spec.comment("\nWriting breakout recording region:\n")
        spec.switch_write_focus(
            BreakoutMachineVertex._BREAKOUT_REGIONS.RECORDING.value)
        ip_tags = tags.get_ip_tags_for_vertex(self) or []
        spec.write_array(recording_utilities.get_recording_header_array(
            [self._recording_size], ip_tags=ip_tags))

        spec.comment("\nWriting breakout param region:\n")
        spec.switch_write_focus(
            BreakoutMachineVertex._BREAKOUT_REGIONS.PARAMS.value)
        spec.write_value(self._x_factor, data_type=DataType.UINT32)
        spec.write_value(self._y_factor, data_type=DataType.UINT32)
        spec.write_value(self._bricking, data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[0], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[1], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[2], data_type=DataType.UINT32)
        spec.write_value(self._rand_seed[3], data_type=DataType.UINT32)

        # End-of-Spec:
        spec.end_specification()

    @property
    def resources_required(self):
        return self._resources_required

    def get_recording_region_base_address(self, txrx, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement, self._BREAKOUT_REGIONS.RECORDING.value, txrx)

    def get_recorded_region_ids(self):
        """ Get the recording region IDs that have been recorded using buffering

        :return: The region numbers that have active recording
        :rtype: iterable(int) """
        return [0]

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "breakout.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE
