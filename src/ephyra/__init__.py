#!/usr/bin/env python
import wxversion

wxversion.ensureMinimal("2.8")

import wx
import os
import logging
import argparse
import cProfile

logging.basicConfig(level = logging.DEBUG)

_ROOT = os.path.abspath(os.path.dirname(__file__))

class EventLoggingApp(wx.PySimpleApp):
	def FilterEvent(self, evt, *args, **kwargs):
		logging.info(evt)
		return -1


def main():
	description = "GUI Simulation and Analysis Suite for the Aietes framework"

	parser = argparse.ArgumentParser(description = description)

	parser.add_argument('-o', '--open',
	                    dest = 'data_file', action = 'store', default = None,
	                    metavar = 'XXX.npz',
	                    help = 'Aietes DataPackage to be analysed'
	)
	parser.add_argument('-a', '--autostart',
	                    dest = 'autostart', action = 'store_true', default = False,
	                    help = 'Automatically launch animation on loading'
	)
	parser.add_argument('-x', '--autoexit',
	                    dest = 'autoexit', action = 'store_true', default = False,
	                    help = 'Automatically exit after animation'
	)
	parser.add_argument('-l', '--loop',
	                    dest = 'loop', action = 'store_true', default = False,
	                    help = 'Loop animation'
	)
	parser.add_argument('-v', '--verbose',
	                    dest = 'verbose', action = 'store_true', default = False,
	                    help = 'Verbose Debugging Information'
	)
	parser.add_argument('-n', '--new-simulation',
	                    dest = 'newsim', action = 'store_true', default = False,
	                    help = 'Generate a new simulation from default'
	)
	args = parser.parse_args()

	if True:
		app = wx.PySimpleApp()
	else:
		app = EventLoggingApp()
		app.SetCallFilterEvent(True)

	from Controller import EphyraController
	from Views import EphyraNotebook

	controller = EphyraController(exec_args = args)
	app.frame = EphyraNotebook(controller, exec_args = args)
	app.frame.Show()
	app.MainLoop()


def debug():
	cProfile.run('main()')

if __name__ == '__main__':
	main()


