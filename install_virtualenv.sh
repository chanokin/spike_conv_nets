#!/bin/bash
if [ $# -eq 0 ]
then
  BRANCH="master"
else
  BRANCH=$1
fi

#---------------------------------------
# Check modules out of git
#---------------------------------------
# Clone low-level tools
git clone -b $BRANCH https://github.com/SpiNNakerManchester/spinnaker_tools.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/spinn_common.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/ybug.git

# Clone sPyNNaker and requirements
git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNUtils.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/DataSpecification.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNMachine.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNMan.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/PACMAN.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNFrontEndCommon.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/sPyNNaker.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNStorageHandlers.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/SpiNNakerGraphFrontEnd.git
# Clone extra modules
git clone -b $BRANCH https://github.com/SpiNNakerManchester/sPyNNakerExtraModelsPlugin.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/sPyNNakerExternalDevicesPlugin.git
git clone -b $BRANCH https://github.com/SpiNNakerManchester/PyNNExamples.git

#---------------------------------------
# Setup virtualenv
#---------------------------------------
# Create virtualenv
virtualenv virtualenv --system-site-packages

# Activate the virtualenv
cd virtualenv
. bin/activate

#---------------------------------------
# Install python modules
#---------------------------------------
# Install python modules
cd ../SpiNNUtils
python setup.py develop --no-deps

cd ../SpiNNMachine
python setup.py develop --no-deps

cd ../DataSpecification
python setup.py develop --no-deps


cd ../SpiNNMan
python setup.py develop --no-deps

cd ../PACMAN
python setup.py develop --no-deps

cd ../SpiNNFrontEndCommon
python setup.py develop --no-deps

cd ../SpiNNakerGraphFrontEnd
python setup.py develop --no-deps

cd ../sPyNNaker
python setup.py develop --no-deps

cd ../sPyNNaker8
python setup.py develop --no-deps

cd ../sPyNNaker8NewModelTemplate
python setup.py develop --no-deps

#cd ../sPyNNakerExternalDevicesPlugin
#python setup.py develop --no-deps

#cd ../sPyNNakerExtraModelsPlugin
#python setup.py develop --no-deps


#---------------------------------------
# Build C
#---------------------------------------
# Source spinnaker tools
cd ../spinnaker_tools
source ./setup
make

# Make spinn_common
cd ../spinn_common
make
make install

cd ../SpiNNFrontEndCommon/c_common
make

cd ../../sPyNNaker/neural_modelling
make

#cd ../../SpiNNMan/c_models/
#make

cd ../../sPyNNaker/neural_modelling/
NEURAL_MODELLING_DIRS=$PWD

cd ../../spinnaker_tools
SPINN_DIRS=$PWD

pip install rig
pip install rig_c_sa

cd ../virtualenv/bin
echo -e "\nexport NEURAL_MODELLING_DIRS=$NEURAL_MODELLING_DIRS" >> activate
echo -e "\nexport SPINN_DIRS=$SPINN_DIRS" >> activate
echo -e "\nexport SPINN_VERSION=131" >> activate

