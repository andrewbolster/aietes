#! /bin/sh
#
# results_sync.sh
# Copyright (C) 2016 bolster <bolster@milo>
#
# Distributed under terms of the MIT license.
#


rsync --info=progress2 -avz ~/src/aietes/results/ bolster@services10.anrg.liv.ac.uk:/volume1/homes/bolster/aietes_results/
rsync --info=progress2 -avz ~/src/aietes/results/ bolster@snowden.home.bolster.online:/data/Documents/results/
