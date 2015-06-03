AIETES
======
Andrew Bolster 2012

[![Code Issues](http://www.quantifiedcode.com/api/v1/project/78cdaccc129f4d878cc319a938186212/badge.svg)](http://www.quantifiedcode.com/app/project/78cdaccc129f4d878cc319a938186212)

##Description
Aietes is a motion, behaviour, and communications simulator for AUVs (Autonomous Underwater Vehicles)
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
* cython

##Installation
    sudo apt-get install dvipng build-essential python-dev python-matplotlib python-simpy python-pydot python-numexpr python-wxgtk2.8 libfreetype6-dev python-tk libhdf5-dev liblapack-dev libblas-dev gfortran
    sudo pip install -r requirements
    python setup.py [install / develop --user]
    python -m unitttests discover
