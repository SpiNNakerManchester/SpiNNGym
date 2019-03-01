#!/bin/bash
curr_dir=`pwd`

cd ../../../../spinnaker/spinnaker_tools
source ./setup

cd ../sPyNNaker/neural_modelling
source ./setup

cd $curr_dir

export TESTING_RESOLUTION=1
make
unset TESTING_RESOLUTION
make
