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
from Simulation import Simulation
import profile
from datetime import datetime as dt

# Uncomment the following section if you want readline history support.
#import readline, atexit
#histfile = os.path.join(os.environ['HOME'], '.TODO_history')
#try:
#    readline.read_history_file(histfile)
#except IOError:
#    pass
#atexit.register(readline.write_history_file, histfile)
def main ():
    """
    Everyone knows what the main does; it does everything!
    """

    global options, args

    outfile=None
    sim = Simulation()

    if options.input is None:
        sim.prepare()
        sim.simulate()

    if options.output:
        print("Storing output in %s"%outfile)
        outfile=dt.now().strftime('%Y-%m-%d-%H-%M-%S.aietes')
        sim.postProcess(inputFile=options.input,outputFile=outfile,dataFile=options.data,movieFile=options.movie, fps=options.fps)

    if options.plot:
        sim.postProcess(inputFile=options.input, displayFrames=720)



if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = optparse.OptionParser(
                formatter=optparse.TitledHelpFormatter(),
                usage=globals()['__doc__'],
                version='$Id: py.tpl 332 2008-10-21 22:24:52Z root $')
        parser.add_option ('-v', '--verbose', action='store_true',
                default=False, help='verbose output')
        parser.add_option ('-P', '--profile', action='store_true',
                default=False, help='profiled execution')
        parser.add_option ('-p', '--plot', action='store_true',
                default=False, help='perform ploting')
        parser.add_option ('-o', '--output', action='store_true',
                default=False, help='store output')
        parser.add_option ('-m', '--movie', action='store_true',
                default=None, help='generate and store movie (this takes a long time)')
        parser.add_option ('-f', '--fps', action='store', type="int",
                default=24, help='set the fps for animation')
        parser.add_option ('-d', '--data', action='store_true',
                default=None, help='store output to datafile')
        parser.add_option ('-i', '--input', action='store', dest='input',
                default=None, help='store input file, this kills the simulation')
        (options, args) = parser.parse_args()
        print options
        #if len(args) < 1:
        #    parser.error ('missing argument')
        if options.verbose: print time.asctime()
        if options.profile:
            profile.run('print("PROFILING"); exit_code=main(); print("EXIT CODE:%s"%exit_code)')
        else:
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
