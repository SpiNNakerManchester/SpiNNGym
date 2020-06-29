import enum
import numpy as np

import matplotlib.animation as animation
import matplotlib.colors as col
import matplotlib.pyplot as plt

BRIGHT_GREEN = (0.0, 0.9, 0.0)


# ----------------------------------------------------------------------------
# InputState
# ----------------------------------------------------------------------------
# Input states
class InputState(enum.IntEnum):
    idle = -1
    right = 0
    left = 1


# ----------------------------------------------------------------------------
# Visualiser
# ----------------------------------------------------------------------------
class Visualiser_subsamp(object):
    # How many bits are used to represent colour
    colour_bits = 0

    def __init__(self, key_input_connection, spike_output_connection,
                 on_pop_name, off_pop_name, x_res=160, y_res=128,
                 x_bits=8, y_bits=8):
        # Reset input state
        self.input_state = InputState.idle

        # Cache reference to key input connection
        self.key_input_connection = key_input_connection
        self.spike_output_connection = spike_output_connection

        # max neuron ID
        self.maxNeuronID = x_res*y_res

        # Build conversion vals
        self.x_res = x_res
        self.y_res = y_res

        # setup output spikes connection callbacks
        self.spike_output_connection.add_receive_callback(
            on_pop_name, self.receive_spikes)
        self.spike_output_connection.add_receive_callback(
            off_pop_name, self.receive_spikes)
        # setup neuron_ids list
        self.neuron_ids_on = []
        self.neuron_ids_off = []

        # Make awesome CRT palette
        cmap = col.ListedColormap(["black", BRIGHT_GREEN])

        # Create image plot to display game screen
        self.fig, self.axis = plt.subplots()
        self.image_data = np.zeros((y_res, x_res, 3))
        self.image = self.axis.imshow(self.image_data, interpolation="nearest",
                                      cmap=cmap, vmin=0.0, vmax=100.0)
        # Hook key listeners
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        # Hide grid
        self.axis.grid(False)
        self.axis.set_xticklabels([])
        self.axis.set_yticklabels([])
        self.axis.axes.get_xaxis().set_visible(False)

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def show(self):
        # Play animation
        self.animation = animation.FuncAnimation(self.fig, self._update,
                                                 interval=20.0, blit=False)
        # Show animated plot (blocking)
        plt.show()

    # spike receiver callback
    def receive_spikes(self, label, time, neuron_ids):
        # add received spike IDs to list
        if label == "subsample channel on":
            for n_id in neuron_ids:
                self.neuron_ids_on.append(np.uint32(n_id))

        elif label == "subsample channel off":
            for n_id in neuron_ids:
                self.neuron_ids_off.append(np.uint32(n_id))

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------
    def _update(self, frame):
        # If state isn't idle, send spike to key input
        if self.input_state != InputState.idle:
            self.key_input_connection.send_spike("key_input", self.input_state)

        if self.neuron_ids_on or self.neuron_ids_off:
            # draw blank background
            self.image_data[:] = 0

        if self.neuron_ids_on:
            payload = np.asarray(self.neuron_ids_on, dtype="uint32")
            # clear neuron_ids list
            self.neuron_ids_on[:] = []

            vision_payload = payload  # & self.value_mask
            # extract coordinates
            x = vision_payload % self.x_res
            y = vision_payload // self.x_res

            # Set valid pixels
            self.image_data[y, x, 1] = 100

        if self.neuron_ids_off:
            payload = np.asarray(self.neuron_ids_off, dtype="uint32")
            # clear neuron_ids list
            self.neuron_ids_off[:] = []

            vision_payload = payload  # & self.value_mask
            # extract coordinates
            x = vision_payload % self.x_res
            y = vision_payload // self.x_res

        # Set valid pixels
        try:
            self.image_data[y, x, 0] = 100
        except IndexError as e:
            print("Packet contains invalid pixels:",
                  vision_payload, x, y, e)

        # Set image data
        self.image.set_array(self.image_data)

        # Return list of artists which we have updated
        # **YUCK** order of these dictates sort order
        # **YUCK** score_text must be returned whether it has
        # been updated or not to prevent overdraw
        return [self.image]

    def _on_key_press(self, event):
        # Send appropriate bits
        if event.key == "left":
            self.input_state = InputState.left
        elif event.key == "right":
            self.input_state = InputState.right

    def _on_key_release(self, event):
        # If either key is released set state to idle
        if event.key == "left" or event.key == "right":
            self.input_state = InputState.idle
