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

ifndef FEC_INSTALL_DIR
    $(error FEC_INSTALL_DIR is not set.  Please define FEC_INSTALL_DIR (possibly by running "source setup_fec" in the sPyNNaker source folder))
endif

APP := inverted_pendulum
SOURCES = inverted_pendulum.c
APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../../spinn_gym/model_binaries/)/
LIBS += -lm

include $(FEC_INSTALL_DIR)/make/fec.mk
