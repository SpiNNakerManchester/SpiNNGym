# Copyright (c) 2019 The University of Manchester
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

DIRS = breakout inverted_pendulum double_inverted_pendulum multi_arm_bandit logic store_recall
#DIRS = breakout double_inverted_pendulum multi_arm_bandit logic store_recall

all: $(DIRS)
	@for d in $(DIRS); do $(MAKE) -C $$d || exit $$?; done

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(abspath $(dir $(MAKEFILE_PATH)))
APP_OUTPUT_DIR := $(CURRENT_DIR)/../spinn_gym/model_binaries/

clean: $(DIRS)
	$(RM) $(APP_OUTPUT_DIR)logs*.sqlite3
	$(RM) $(APP_OUTPUT_DIR)*.aplx
	@for d in $(DIRS); do $(MAKE) -C $$d GNU=$(GNU) clean || exit $$?; done
    