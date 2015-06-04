Aietes
======
Andrew Bolster 2012-2015

#Description
Aietes is a motion, behaviour, and communications simulator for AUVs (Autonomous Underwater Vehicles)

#Dependencies
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

#Installation
    sudo apt-get install dvipng build-essential python-dev python-matplotlib python-simpy python-pydot python-numexpr python-wxgtk2.8 libfreetype6-dev python-tk libhdf5-dev liblapack-dev libblas-dev gfortran
    sudo pip install -r requirements
    python setup.py [install / develop --user]
    python -m unitttests discover

# Source Maintance 

## Formatting 

[![Code Issues](http://www.quantifiedcode.com/api/v1/project/78cdaccc129f4d878cc319a938186212/badge.svg)](http://www.quantifiedcode.com/app/project/78cdaccc129f4d878cc319a938186212)

PyCharms auto formatter sucks for PEP8

    autopep8 -ivr -j 4 --exclude Ghia --ignore E501 --max-line-length 119 src scripts   

## Testing

Ghia tests are reasonably deprecated but are still ran if that module is available, and they . No harm in it really.


