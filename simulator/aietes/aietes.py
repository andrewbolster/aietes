#!/usr/bin/env python

"""
SYNOPSIS

    TODO helloworld [-h] [-v,--verbose] [--version]

DESCRIPTION

    TODO This describes how to use this script.
    This docstring will be printed by the script if there is an error or
    if the user requests help (-h or --help).

EXAMPLES

    TODO: Show some examples of how to use this script.

EXIT STATUS

    TODO: List exit codes

AUTHOR

    TODO: Name <name@example.org>

LICENSE

    This script is in the public domain.

VERSION

    
"""

import sys
import os
import traceback
import optparse
import time
#from pexpect import run, spawn

from SimPy.Simulation import *
from SimPy.SimPlot import *

# Uncomment the following section if you want readline history support.
#import readline, atexit
#histfile = os.path.join(os.environ['HOME'], '.TODO_history')
#try:
#    readline.read_history_file(histfile)
#except IOError:
#    pass
#atexit.register(readline.write_history_file, histfile)
class Behaviour():
    """
    Represents a given behaviour
    """

class Environment():
    """
    Represents the real environment within which the simulation runs
    """
    size= 1000000
    dim= pow(size,-3)
    xmin=-dim
    xmax=dim
    ymin=-dim
    ymax=dim
    zmin=-dim
    zmax=dim


class Node(Process):
    """
    Represents individual craft/nodes/vectors, including behaviour, comms, etc
    """
    fleet=None
    # Position and Orientation as a single vector #[x,y,z,alpha, beta, gamma]
    placement = [ 0, 0, 0, 0, 0, 0 ]

    def __init__(self,fleet,node_def):
        self.fleet = fleet
        self.id = fleet.give_id()

    def _move(self,force_vector):
        """
        Allow movement with either a 3- or 6- vector
        """
        if len(vector) == 3:
            placement[:3]+=force_vector
        elif len(vector) == 6:
            placement+=force_vector
        else
            raise ValueError("Incorrect Vector Definition")

    def 

def Fleet():
    """
    Encapsulates fleet level characteristics and behaviours, inc node 
    generation etc
    """
    nodes=[]
    def __init__(self,fleet_def):
        for node_def in fleet_def.node_defs:
            self.nodes.append(Node(self,node_def))


def SimRun(Simulation)
    """
    Encapsulates an individual simulation execution, including all relevant 
    contextual information to provide replication
    """
    fleets=[]
    def __init__(self,context)

def main ():
    """
    Everyone knows what the main does; it does everything!
    """

    global options, args

    if options.context_file:
        # TODO Import context
    else
        context= Context()

    sim = Simulation()

    sim.run()





if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = optparse.OptionParser(
                formatter=optparse.TitledHelpFormatter(),
                usage=globals()['__doc__'],
                version='$Id: py.tpl 332 2008-10-21 22:24:52Z root $')
        parser.add_option ('-v', '--verbose', action='store_true',
                default=False, help='verbose output')
        (options, args) = parser.parse_args()
        #if len(args) < 1:
        #    parser.error ('missing argument')
        if options.verbose: print time.asctime()
        exit_code = main()
        if exit_code is None:
            exit_code = 0
        if options.verbose: print time.asctime()
        if options.verbose: print 'TOTAL TIME IN MINUTES:',
        if options.verbose: print (time.time() - start_time) / 60.0
        sys.exit(exit_code)
    except KeyboardInterrupt, e: # Ctrl-C
        raise e
    except SystemExit, e: # sys.exit()
        raise e
    except Exception, e:
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)

# vim:set sr et ts=4 sw=4 ft=python fenc=utf-8: // See Vim, :help 'modeline'
