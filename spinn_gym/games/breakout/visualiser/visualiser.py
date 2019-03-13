import multiprocessing

import enum
import numpy as np
import socket

import matplotlib.animation as animation
import matplotlib.colors as col
import matplotlib.pyplot as plt
import datetime

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import io
import os

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

    def __init__(self, udp_port, key_input_connection=None, scale=4,
                 x_factor=8, y_factor=8, x_bits=8, y_bits=8, fps=60):
        # Reset input state
        self.input_state = InputState.idle

        # Zero score
        self.score = 0

        # Cache reference to key input connection
        self.key_input_connection = key_input_connection

        # Build masks
        self.x_mask = (1 << x_bits) - 1
        self.x_shift = self.colour_bits + y_bits + 1
        # assert self.x_shift == 10, self.x_shift

        self.y_mask = (1 << y_bits) - 1
        self.y_shift = self.colour_bits

        # assert self.y_shift == 2, self.y_shift

        self.colour_mask = 1
        self.bricked_mask = 2

        self.value_mask = (1 << (x_bits + y_bits + self.colour_bits)) - 1

        # assert x_bits + y_bits + self.colour_bits == 18, x_bits + y_bits + self.colour_bits

        # assert self.value_mask == 0x3FFFF, self.value_mask

        self.y_res = 128 / y_factor
        self.x_res = 160 / x_factor
        self.BRICK_WIDTH = self.x_res / (32 / x_factor)
        self.BRICK_HEIGHT = 32 / y_factor
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.bat_width = 32 / x_factor
        self.fps = fps
        self.scale = scale

        print "x_factor", self.x_factor
        print "y_factor", self.y_factor
        print "x_res", self.x_res
        print "y_res", self.y_res
        print "x_bits", x_bits
        print "y_bits", y_bits
        print "x_mask", self.x_mask
        print "y_mask", self.y_mask
        print "v_mask", self.value_mask
        print "bat width", self.bat_width
        print "Brick Width", self.BRICK_WIDTH
        print "Brick Height", self.BRICK_HEIGHT

        # Open socket to receive datagrams
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", udp_port))
        self.socket.setblocking(False)

        # Make awesome CRT palette
        cmap = col.ListedColormap(["black", BRIGHT_GREEN, BRIGHT_RED, BRIGHT_PURPLE, BRIGHT_BLUE, BRIGHT_ORANGE])

        # Create image plot to display game screen
        self.fig = plt.figure("BreakOut", figsize=(8, 6))
        self.axis = plt.subplot(1, 1, 1)
        # self.ion = plt.ion()
        self.image_data = np.zeros((self.y_res, self.x_res))
        self.image = self.axis.imshow(self.image_data, interpolation="nearest",
                                      cmap=cmap, vmin=0.0, vmax=5.0)

        # Draw score using textbox
        self.score_text = self.axis.text(0.5, 1.0, "0", color=BRIGHT_GREEN,
                                         transform=self.axis.transAxes,
                                         horizontalalignment="right",
                                         verticalalignment="top")
        # Hook key listeners
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        # Hide grid
        self.axis.grid(False)
        self.axis.set_xticklabels([])
        self.axis.set_yticklabels([])
        self.axis.axes.get_xaxis().set_visible(False)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_data = np.zeros((self.y_res, self.x_res, 3), dtype='uint8')
        # self.video_data = cv2.imread("temp_frame.png")
        self.video_shape = (self.x_res * self.scale, self.y_res * self.scale)
        self.dsize = (self.y_res * self.scale, self.x_res * self.scale)

        filename = os.path.join(os.getcwd(), "breakout_output_%s.m4v" %
                    datetime.datetime.now().strftime("%Y-%m-%d___%H-%M-%S"))
        # print filename
        self.video_writer = cv2.VideoWriter(
            filename,
            fourcc, self.fps,
            self.video_shape,
            isColor=True)
        self.video_writer.open(filename,
            fourcc, self.fps,
            self.video_shape,
            isColor=True)

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def show(self):
        # Play animation
        interval = (1000. / self.fps)
        self.animation = animation.FuncAnimation(self.fig, self._update,
                                                 interval=interval,
                                                 blit=False)
        # Show animated plot (blocking)
        try:
            plt.show()
        except:
            pass

    def handle_close(self, evt):
        self.video_writer.release()

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------
    def _update(self, frame):
        # print "trying to update interval = ", (1000. / self.fps)

        # If state isn't idle, send spike to key input
        if self.input_state != InputState.idle and self.key_input_connection:
            self.key_input_connection.send_spike("key_input", self.input_state)

        # Read all datagrams received during last frame
        message_received = False
        while True:
            try:
                raw_data = self.socket.recv(512)
            except socket.error:
                # If error isn't just a non-blocking read fail, print it
                # if e != "[Errno 11] Resource temporarily unavailable":
                #    print "Error '%s'" % e
                # Stop reading datagrams
                break
            else:
                message_received = True
                # Slice off EIEIO header and convert to numpy array of uint32
                payload = np.fromstring(raw_data[6:], dtype="uint32")

                payload_value = payload & self.value_mask
                vision_event_mask = payload_value >= SpecialEvent.max
                # Payload is a pixel:
                # Create mask to select vision (rather than special event) packets

                # Extract coordinates
                'const uint32_t spike_key = ' \
                    'key | (SPECIAL_EVENT_MAX + (i << (game_bits + 2)) + (j << 2) + (bricked<<1) + colour_bit);'
                vision_payload = payload_value[
                                     vision_event_mask] - SpecialEvent.max
                x = (vision_payload >> self.x_shift) & self.x_mask
                y = (vision_payload >> self.y_shift) & self.y_mask

                c = (vision_payload & self.colour_mask)
                b = (vision_payload & self.bricked_mask) >> 1

                '''if y.any() == self.y_res-1:
                    if c[np.where(y==self.y_res-1)].any()==1:
                        #add remaining bat pixels to image
                        x_pos=x[np.where(y==self.y_res-1)]
                        for i in range(1,self.bat_width):
                            np.hstack((y,self.y_res-1))
                            np.hstack((c,1))
                            np.hstack((x,x_pos+i))'''
                # Set valid pixels
                try:
                    for x1, y1, c1, b1 in zip(x, y, c, b):
                        # self.image_data[:] = 0
                        print "valid pixels = x:{}\ty:{}\tc:{}\tb:{}".format(x, y, c, b)
                        if b1 == 0:
                            self.image_data[y1, x1] = c1
                        elif b1 == 1:
                            self.image_data[y1:(y1 + self.BRICK_HEIGHT),
                            x1:(x1 + self.BRICK_WIDTH)] = c1 * np.random.randint(2, 6)

                    # if c>0:
                    # self.video_data[:] = 0
                    self.video_data[y, x, 1] = np.uint8(c * 230)
                    # else:
                    #     self.video_data[y, x, :] = VIDEO_RED
                except IndexError as e:
                    print("Packet contains invalid pixels:",
                          vision_payload, "X:", x, "  Y:", y, " c:", c, " b:",
                          b)
                    # self.image_data[:-1, :] = 0

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

                if self.score > 0:
                    # print("pos score %d"%self.score)
                    self.video_data[0:1, :, :] = [100, 255, 255]
                elif self.score < 0:
                    # print("neg score %d"%self.score)
                    self.video_data[0:1, :, :] = [255, 100, 255]
                else:
                    # print("score 0")
                    self.video_data[0:1, :, :] = [200, 200, 200]

        # Set image data
        try:
            self.image.set_array(self.image_data)
        except NameError:
            pass

        # try:
        if message_received:
            #     if not self.first_update:
            #         buf = io.BytesIO()
            #         plt.savefig(buf, format='png', dpi=100)
            #         buf.seek(0)
            #         self.video_data[:] = cv2.imdecode(
            #             np.fromstring(buf.read(), np.uint8),
            #             cv2.IMREAD_COLOR)
            #     self.video_writer.write(self.video_data)

            self.video_writer.write(
                cv2.resize(self.video_data, self.video_shape,
                           interpolation=cv2.INTER_NEAREST))
        # except:
        #     pass

        # Return list of artists which we have updated
        # **YUCK** order of these dictates sort order
        # **YUCK** score_text must be returned whether it has
        # been updated or not to prevent overdraw
        # self.first_update = False
        return [self.image, self.score_text]

    def _on_key_press(self, event):
        # Send appropriate bits
        if event.key == "left":
            # print
            self.input_state = InputState.left
        elif event.key == "right":
            self.input_state = InputState.right

    def _on_key_release(self, event):
        # If either key is released set state to idle
        if event.key == "left" or event.key == "right":
            self.input_state = InputState.idle


if __name__ == "__main__":
    print("Running from command line")
    UDP_PORT = 17893
    vis = Visualiser(UDP_PORT)
    vis.show()

