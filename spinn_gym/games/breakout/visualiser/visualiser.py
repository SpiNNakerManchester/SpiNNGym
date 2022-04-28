# Copyright (c) 2019-2021 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import enum
import numpy as np

import matplotlib.colors as col
import matplotlib.pyplot as plt
import datetime
import time

import cv2
import os
import sys

from collections import deque

BRIGHT_GREEN = (0.0, 0.9, 0.0)
BRIGHT_RED = (0.9, 0.0, 0.0)
BRIGHT_BLUE = (0, 0.0, 0.9)
BRIGHT_PURPLE = (0.9, 0.0, 0.9)
BRIGHT_ORANGE = (0.9, 0.4, 0.0)
VIDEO_GREEN = np.array([0, 230, 0])
VIDEO_RED = np.array([230, 0, 0])


# ----------------------------------------------------------------------------
# InputState
# ----------------------------------------------------------------------------
# Input states
class InputState(enum.IntEnum):
    idle = -1
    left = 0
    right = 1


# ----------------------------------------------------------------------------
# SpecialEvent
# ----------------------------------------------------------------------------
# Special events sent from game using first keys
class SpecialEvent(enum.IntEnum):
    score_up = 0
    score_down = 1
    max = 2


# ----------------------------------------------------------------------------
# Visualiser
# ----------------------------------------------------------------------------
class Visualiser(object):
    # How many bits are used to represent colour and brick
    colour_bits = 2

    def __init__(self, key_input_connection=None,
                 scale=4, x_factor=8, y_factor=8, x_bits=8, y_bits=8, fps=60,
                 live_pops=None, live_duration=1000, video_out=False):
        self._connection_ready = False
        self.running = True
        self.message_received = False

        # Reset input state
        self.input_state = InputState.idle

        # Zero score
        self.score = 0

        # Cache reference to key input connection
        self.key_input_connection = key_input_connection

        # Build masks
        self.x_mask = (1 << x_bits) - 1
        self.x_shift = self.colour_bits + y_bits
        # assert self.x_shift == 10, self.x_shift

        self.y_mask = (1 << y_bits) - 1
        self.y_shift = self.colour_bits

        # assert self.y_shift == 2, self.y_shift
        self.colour_mask = 1
        self.bricked_mask = 2

        self.value_mask = (1 << (x_bits + y_bits + self.colour_bits)) - 1

        self.y_res = int(128 / y_factor)
        self.x_res = int(160 / x_factor)
        self.BRICK_WIDTH = int(self.x_res / 5)
        self.BRICK_HEIGHT = int(16 / y_factor)
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.bat_width = int(32 / x_factor)
        self.fps = fps
        self.scale = scale

        print("\n\nVisualiser Initialised With Parameters:")
        print("\tx_factor {}".format(self.x_factor))
        print("\ty_factor {}".format(self.y_factor))
        print("\tx_res {}".format(self.x_res))
        print("\ty_res {}".format(self.y_res))
        print("\tx_bits {}".format(x_bits))
        print("\ty_bits {}".format(y_bits))
        print("\tx_mask {}".format(self.x_mask))
        print("\ty_mask {}".format(self.y_mask))
        print("\tv_mask {}".format(self.value_mask))
        print("\tbat width {}".format(self.bat_width))
        print("\tBrick Width {}".format(self.BRICK_WIDTH))
        print("\tBrick Height {}".format(self.BRICK_HEIGHT))

        # Make awesome CRT palette
        cmap = col.ListedColormap(["black", BRIGHT_GREEN, BRIGHT_RED,
                                   BRIGHT_PURPLE, BRIGHT_BLUE, BRIGHT_ORANGE])

        # Create image plot to display game screen
        axes_names = [["Breakout"]]
        width = 8
        if live_pops:
            width = 16
            breakouts = ["Breakout"] * len(live_pops)
            axes_names = [["Breakout", pop.label] for pop in live_pops]
        
        self.fig, self.axes = plt.subplot_mosaic(
            axes_names, figsize=(width, 6), constrained_layout=True)
        
        if live_pops:
            self.live_spike_data = {pop.label: deque() for pop in live_pops}
            self.live_spike_range = dict()
            self.live_spike_plot = dict()
            for pop in live_pops:
                self.live_spike_plot[pop.label] = self.axes[pop.label].plot(
                    [], [], ".")
                self.axes[pop.label].set_xlim(0, live_duration)
                self.axes[pop.label].set_ylim(0, pop.size)
                self.axes[pop.label].set_xticks([0, live_duration - 100])
                self.live_spike_range[pop.label] = (0, live_duration)
            self.live_duration = live_duration
        
        breakout_axis = self.axes["Breakout"]
        self.image_data = np.zeros((self.y_res, self.x_res))
        self.image = breakout_axis.imshow(self.image_data, interpolation="nearest",
                                      cmap=cmap, vmin=0.0, vmax=5.0)

        # Draw score using textbox
        self.score_text = breakout_axis.text(
            0.5, 1.0, "Waiting for simulation to start...",
            color=BRIGHT_GREEN, transform=breakout_axis.transAxes,
            horizontalalignment="right", verticalalignment="top")

        # Hook key listeners
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        
        # Hide grid
        breakout_axis.grid(False)
        breakout_axis.set_xticklabels([])
        breakout_axis.set_yticklabels([])
        breakout_axis.axes.get_xaxis().set_visible(False)
        
        if video_out:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_data = np.zeros((self.y_res, self.x_res, 3), dtype='uint8')
            self.video_shape = (self.x_res * self.scale, self.y_res * self.scale)
            self.dsize = (self.y_res * self.scale, self.x_res * self.scale)

            filename = os.path.join(os.getcwd(), "breakout_output_%s.m4v" %
                                    datetime.datetime.now().strftime(
                                        "%Y-%m-%d___%H-%M-%S"))
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, self.fps, self.video_shape, isColor=True)
            self.video_writer.open(
                filename, fourcc, self.fps, self.video_shape, isColor=True)
        else:
            self.video_data = None

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def show(self):
        # Show animated plot (blocking)
        try:
            plt.ion()
            plt.show()
            plt.draw()
            print("Visualiser displayed")
        except Exception:
            pass

    def handle_close(self, evt):
        self.close()

    def close(self):
        self.score_text.set_text("Simulation finished")
        self.running = False
        self.video_writer.release()
        
    def handle_live_spikes(self, label, time, neuron_ids):
        # Take out the trash
        time_low, time_high = self.live_spike_range[label]
        if time < time_low:
            return
        if time > time_high:
            time_high = time
            time_low = time - self.live_duration
            self.live_spike_range[label] = (time_low, time_high)
            self.axes[label].set_xlim(time_low, time_high)
            self.axes[label].set_xticks([time_low, time_high - 100])
        
    def handle_breakout_spikes(self, neuron_ids):
        payload = np.array(neuron_ids, dtype="uint32")
        payload_value = payload & self.value_mask
        vision_event_mask = payload_value >= SpecialEvent.max

        # Payload is a pixel:

        # Create mask to select vision (rather than event) packets
        # Extract coordinates
        'const uint32_t spike_key = ' \
            'key | (SPECIAL_EVENT_MAX + ' \
            '(i << (game_bits + 2)) + (j << 2) + ' \
            '(bricked<<1) + colour_bit);'

        vision_payload = payload_value[
                             vision_event_mask] - SpecialEvent.max
        x = (vision_payload >> self.x_shift) & self.x_mask
        y = (vision_payload >> self.y_shift) & self.y_mask

        c = (vision_payload & self.colour_mask)
        b = (vision_payload & self.bricked_mask) >> 1

        # Set valid pixels
        try:
            for x1, y1, c1, b1 in zip(x, y, c, b):

                if b1 == 0:
                    self.image_data[y1, x1] = c1

                elif b1 == 1:
                    self.image_data[y1:(y1 + self.BRICK_HEIGHT),
                                    x1:(x1 + self.BRICK_WIDTH)] = c1

            if self.video_data is not None:
                self.video_data[y, x, 1] = np.uint8(c * 230)
        except IndexError as e:
            print("Packet contains invalid pixels:",
                  vision_payload, "X:", x, "  Y:", y, " c:", c, " b:",
                  b, e)

        # Create masks to select score events and count them
        num_score_up_events = np.sum(
            payload_value == SpecialEvent.score_up)
        num_score_down_events = np.sum(
            payload_value == SpecialEvent.score_down)

        # If any score events occurred
        if num_score_up_events > 0 or num_score_down_events > 0:
            # Apply to score count
            self.score += num_score_up_events
            self.score -= num_score_down_events

            # Update displayed score count
            self.score_text.set_text("%u" % self.score)
            
        if self.video_data is not None:
            if self.score > 0:
                # print("pos score %d"%self.score)
                self.video_data[0:1, :, :] = [100, 255, 255]
            elif self.score < 0:
                # print("neg score %d"%self.score)
                self.video_data[0:1, :, :] = [255, 100, 255]
            else:
                # print("score 0")
                self.video_data[0:1, :, :] = [200, 200, 200]
        self.message_received = True

    def update(self, frame):
        # If state isn't idle, send spike to key input
        if self.input_state != InputState.idle and self.key_input_connection:
            self.key_input_connection.send_spike("key_input", self.input_state)

        # Set image data
        try:
            self.image.set_array(self.image_data)
        except NameError:
            pass

        # try:
        if self.message_received and self.video_data is not None:
            self.video_writer.write(
                cv2.resize(self.video_data, self.video_shape,
                           interpolation=cv2.INTER_NEAREST))
            self.message_received = False
        return [self.image, self.score_text]

    def _on_key_press(self, event):
        # Send appropriate bits
        if event.key == "left":
            print("Left key pressed!\n")
            self.input_state = InputState.left
        elif event.key == "right":
            print("Right key pressed!\n")
            self.input_state = InputState.right

    def _on_key_release(self, event):
        print("Key released!\n")
        # If either key is released set state to idle
        if event.key == "left" or event.key == "right":
            self.input_state = InputState.idle
