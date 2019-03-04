ifndef NEURAL_MODELLING_DIRS
    $(error NEURAL_MODELLING_DIRS is not set.  Please define NEURAL_MODELLING_DIRS (possibly by running "source setup" in the neural_modelling folder within the sPyNNaker source folder))
endif

APP := bandit
BUILD_DIR = $(abspath $(CURRENT_DIR)/../../build/$(APP))/
SOURCES = band.c
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(dir $(MAKEFILE_PATH))
EXTRA_SRC_DIR := $(abspath $(CURRENT_DIR))
SOURCE_DIRS += $(EXTRA_SRC_DIR)
APP_OUTPUT_DIR := $(abspath $(CURRENT_DIR)../../../spinn_gym/model_binaries/)/
CFLAGS += -I$(NEURAL_MODELLING_DIRS)/src

include $(SPINN_DIRS)/make/local.mk