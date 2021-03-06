#!/usr/bin/env python
# coding=utf-8
"""
 * This file is part of the Aietes Framework (https://github.com/andrewbolster/aietes)
 *
 * (C) Copyright 2013 Andrew Bolster (http://andrewbolster.info/) and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Andrew Bolster, Queen's University Belfast (-Aug 2013), University of Liverpool (Sept 2014-)
"""
__author__ = "Andrew Bolster"
__license__ = "EPL"
__email__ = "me@andrewbolster.info"

import os
import sys

if not "wx" in sys.modules:
    import wxversion

    wxversion.ensureMinimal("2.8")

import wx
import logging
import argparse
import cProfile
import traceback
from aietes.Tools import get_latest_aietes_datafile, is_valid_aietes_datafile

logging.basicConfig(level=logging.DEBUG)

_ROOT = os.path.abspath(os.path.dirname(__file__))


class EventLoggingApp(wx.PySimpleApp):
    @staticmethod
    def filter_event(evt, *args, **kwargs):
        logging.info(evt)
        return -1


def show_error():
    message = ''.join(traceback.format_exception(*sys.exc_info()))
    dialog = wx.MessageDialog(None, message, 'Error!', wx.OK | wx.ICON_ERROR)
    dialog.ShowModal()


def main():
    description = "GUI Simulation and Analysis Suite for the Aietes framework"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-o', '--open',
                        dest='data_file', action='store', default=None,
                        nargs='?', const='latest_aietes.npz_from_pwd',
                        metavar='XXX.npz',
                        help='Aietes DataPackage to be analysed'
                        )
    parser.add_argument('-a', '--autostart',
                        dest='autostart', action='store_true', default=False,
                        help='Automatically launch animation on loading'
                        )
    parser.add_argument('-x', '--autoexit',
                        dest='autoexit', action='store_true', default=False,
                        help='Automatically exit (after animation with -a)'
                        )
    parser.add_argument('-l', '--loop',
                        dest='loop', action='store_true', default=False,
                        help='Loop animation'
                        )
    parser.add_argument('-v', '--verbose',
                        dest='verbose', action='store_true', default=False,
                        help='Verbose Debugging Information'
                        )
    parser.add_argument('-n', '--new-simulation',
                        dest='newsim', action='store_true', default=False,
                        help='Generate a new simulation from default'
                        )
    args = parser.parse_args()

    if args.data_file is 'latest_aietes.npz_from_pwd':
        args.data_file = get_latest_aietes_datafile()
    elif args.data_file is not None:
        if not is_valid_aietes_datafile(args.data_file):
            raise ValueError(
                "Provided data file does not appear to be an aietes dataset:{0!s}".format(args.data_file))

    logging.info("Using Latest AIETES file: {0!s}".format(args.data_file))

    if True:
        app = wx.PySimpleApp()
    else:
        app = EventLoggingApp()
        app.SetCallFilterEvent(True)

    from Controller import EphyraController
    from Views import EphyraNotebook

    controller = EphyraController(exec_args=args)
    app.frame = EphyraNotebook(controller, exec_args=args)
    app.frame.Show()
    try:
        app.MainLoop()
    except Exception as e:
        raise


def debug():
    cProfile.run('main()')


if __name__ == '__main__':
    main()
