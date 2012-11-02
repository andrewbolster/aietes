#!/usr/bin/env python
"""
AIETES
In Greek Mythology, Aietes was son of the sun-god Helios and the Oceanid Perseis, brother to Circe and Pasiphae, and King of Colchis, and was the King that Jason (of Argonaut fame) acquired the Golden Fleese from, after completing a few tasks, mostly involving dragons.

Oh, and Jason stole his daughter, and killed (and diced) his son. Good times.
"""

import sys
import os
import traceback
import optparse
import time
from aietes import Simulation
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
    sim = Simulation(config_file=options.config)

    if options.input is None:
        sim.prepare()
        sim.simulate()

    if options.output:
        outfile=dt.now().strftime('%Y-%m-%d-%H-%M-%S.aietes')
        print("Storing output in %s"%outfile)
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
        parser.add_option ('-c', '--config', action='store', dest='config',
                default='', help='generate simulation from config file')
        (options, args) = parser.parse_args()
        print options
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
