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

ifndef NEURAL_MODELLING_DIRS
    $(error NEURAL_MODELLING_DIRS is not set.  Please define 
    NEURAL_MODELLING_DIRS (possibly by running "source setup" in the 
    neural_modelling folder within the sPyNNaker source folder))
endif

APP := breakout
BUILD_DIR = $(abspath $(CURRENT_DIR)/../../build/$(APP))/
SOURCES = bkout.c
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(dir $(MAKEFILE_PATH))
EXTRA_SRC_DIR := $(abspath $(CURRENT_DIR))/src
SOURCE_DIRS += $(EXTRA_SRC_DIR)
APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../../spinn_gym/model_binaries/)/

include $(SPINN_DIRS)/make/local.mk

CFLAGS += -I$(NEURAL_MODELLING_DIRS)/src
