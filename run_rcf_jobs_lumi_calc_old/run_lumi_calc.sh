#!/bin/bash
#
# condor executes runana.cmd 0, runana.cmd 1, etc.
# where $1 = 0, 1, etc...
#

export USER="$(id -u -n)"
export LOGNAME=${USER}
export HOME=/sphenix/u/${LOGNAME}

source ${HOME}/.bashrc
source ${HOME}/Software/dylan_env/bin/activate

#print the environment - needed for debugging
printenv


if [ "$condor" = false ]; then
    #=================== Run standalone =========================#
    python ../luminosity_calculation_rcf.py "$1"
    #=========================================================#

else
    #=================== Run with condor =========================#
    python ../luminosity_calculation_rcf.py "$1"
    #=========================================================#
fi
