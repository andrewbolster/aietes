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

import logging

from matplotlib import animation as mplanimation


class AIETESAnimation(mplanimation.FuncAnimation):
    def save(self, filename, fps=5, codec='libx264', clear_temp=True,
             frame_prefix='_tmp', blit=False, *args, **kwargs):
        """
        Saves a movie file by drawing every frame.


        :param args:
        :param kwargs:
        :param blit:
        :param frame_prefix:
        :param clear_temp:
        :param codec:
        :param fps:
        :param filename:
        *filename* is the output filename, eg :file:`mymovie.mp4`

        *fps* is the frames per second in the movie

        *codec* is the codec to be used,if it is supported by the output method.

        *clear_temp* specifies whether the temporary image files should be
        deleted.

        *frame_prefix* gives the prefix that should be used for individual
        image files.  This prefix will have a frame number (i.e. 0001) appended
        when saving individual frames.
        """
        # Need to disconnect the first draw callback, since we'll be doing
        # draws. Otherwise, we'll end up starting the animation.
        if self._first_draw_id is not None:
            self._fig.canvas.mpl_disconnect(self._first_draw_id)
            reconnect_first_draw = True
        else:
            reconnect_first_draw = False

        fnames = []
        # Create a new sequence of frames for saved data. This is different
        # from new_frame_seq() to give the ability to save 'live' generated
        # frame information to be saved later.
        # TODO: Right now, after closing the figure, saving a movie won't
        # work since GUI widgets are gone. Either need to remove extra code
        # to allow for this non-existant use case or find a way to make it
        # work.
        for idx, data in enumerate(self.new_saved_frame_seq()):
            # TODO: Need to see if turning off blit is really necessary
            self._draw_next_frame(data, blit=blit)
            fname = '{0!s}{1:04d}.png'.format(frame_prefix, idx)
            fnames.append(fname)
            self._fig.savefig(fname)

        self._make_movie(
            filename, fps, codec, frame_prefix, cmd_gen=self.mencoder_cmd)

        # Delete temporary files
        if clear_temp:
            import os

            for fname in fnames:
                os.remove(fname)

        # Reconnect signal for first draw if necessary
        if reconnect_first_draw:
            self._first_draw_id = self._fig.canvas.mpl_connect('draw_event',
                                                               self._start)

    @staticmethod
    def ffmpeg_cmd(fname, fps, codec, frame_prefix):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie
        return ['ffmpeg', '-y', '-r', str(fps),
                '-b', '1800k', '-i', '{0!s}%04d.png'.format(frame_prefix),
                '-vcodec', codec, '-vpre', 'slow', '-vpre', 'baseline',
                "{0!s}.mp4".format(fname)]

    @staticmethod
    def mencoder_cmd(fname, fps, codec, frame_prefix):
        # Returns the command line parameters for subprocess to use
        # mencoder to create a movie
        return ['mencoder',
                '-nosound',
                '-quiet',
                '-ovc', 'lavc',
                '-lavcopts', "vcodec={0!s}".format(codec),
                '-o', "{0!s}.mp4".format(fname),
                '-mf', 'type=png:fps=24', 'mf://{0!s}%04d.png'.format(frame_prefix)]

    def _make_movie(self, fname, fps, codec, frame_prefix, cmd_gen=None):
        # Uses subprocess to call the program for assembling frames into a
        # movie file.  *cmd_gen* is a callable that generates the sequence
        # of command line arguments from a few configuration options.
        from subprocess import Popen, PIPE

        if cmd_gen is None:
            cmd_gen = self.ffmpeg_cmd
        command = cmd_gen(fname, fps, codec, frame_prefix)
        print command
        try:
            proc = Popen(command, shell=False,
                         stdout=PIPE, stderr=PIPE)
            proc.wait()
        except OSError:
            logging.critical(
                "Mencoder probably not found in path, try installing it")
            raise
