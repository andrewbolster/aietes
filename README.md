aietes
======
# AIETES
Andrew Bolster 2012
##Description
Aietes is a motion and behaviour simulator for AUVs (Autonomous Underwater Vehicles)
##Dependencies
* matplotlib
* numpy
* scipy
* simpy
* pydot
* python-simpy
* python-pydot
* mencoder #for animation storage
* python-wxgtk2.8
* python-tk
##Installation
    sudo apt-get install build-essential python-dev python-matplotlib python-simpy python-pydot python-numexpr python-wxgtk2.8 libfreetype6-dev python-tk libhdf5-dev liblapack-dev libblas-dev gfortran
    sudo pip install -r requirements
    python setup.py [install / develop --user]
    python -m unitttests discover
