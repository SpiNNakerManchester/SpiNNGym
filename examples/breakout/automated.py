# Copyright (c) 2023 The University of Manchester
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from threading import Thread
import pyNN.spiNNaker as p

from spinn_gym.games.breakout.visualise_host import (
    host_visualiser, start_visualiser)
from spinn_gym.games.breakout.automated_breakout import (
    AutomatedBreakout, X_RES, X_SCALE, Y_RES, Y_SCALE)

# ---------------------------------------------------------------------
# Initialise Simulation and Parameters
# ---------------------------------------------------------------------
run_time_seconds = 60
breakout = AutomatedBreakout(time_scale_factor=2)

# ---------------------------------------------------------------------
# Configure Visualiser
# ---------------------------------------------------------------------
vis = host_visualiser(
    breakout, X_RES, X_SCALE, Y_RES, Y_SCALE,
    [breakout.paddle_pop, breakout.ball_pop,
     breakout.left_hidden_pop, breakout.right_hidden_pop,
     breakout.decision_input_pop])
vis.show()


# --------------------------------------------------------------------
# Run Simulation
# --------------------------------------------------------------------
def do_run():
    print("\nLet\'s play breakout!")
    p.external_devices.run_forever()
    print("Simulation Complete")
    p.end()


thread = Thread(target=do_run)
thread.start()

start_visualiser(vis)
p.external_devices.request_stop()
