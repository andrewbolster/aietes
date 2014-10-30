#!/usr/bin/env python
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

"""
BOUNOS - Heir to the Kingdom of AIETES
"""

import sys
import os
import argparse
from argparse import RawTextHelpFormatter
from math import ceil
from joblib import Parallel, delayed
from natsort import natsorted
import collections
from copy import deepcopy

import numpy as np
import pandas as pd

import Metrics
import Analyses
from DataPackage import DataPackage

from aietes.Tools import list_functions, mkpickle

from pprint import pformat

font = {'family': 'normal',
        'weight': 'normal',
        'size': 10}
from matplotlib.ticker import FuncFormatter

plot_alpha = 0.9

_metrics = [Metrics.Deviation_Of_Heading,
            Metrics.PerNode_Speed,
            Metrics.PerNode_Internode_Distance_Avg]


class BounosModel(DataPackage):

    """
    BounosModel acts as an interactive superclass of DataPackage, designed for interactive
        simulation/analysis

    It is blankly initialised with no arguments and must be initialised by either interacting
        with a simulation (update_data_from_sim) or from an existing datafile (import_datafile)
    """

    def __init__(self, *args, **kwargs):
        self.metrics = []
        self.is_ready = False
        self.is_simulating = None

    def import_datafile(self, file):
        super(BounosModel, self).__init__(source=file)
        self.is_ready = True
        self.is_simulating = False

    def update_data_from_sim(self, p, v, names, environment, now):
        """
        Call back function used by SimulationStep if doing real time simulation

        Imports DataPackage data from the running simulation up to the requested time (self.t)
        """
        self.log.debug("Updating data from simulator at %d" % now)
        self.update(p=p, v=v, names=names, environment=environment)


args = None


def load_sources(sources):
    """
    From a given list of DataPackage-able sources, parallelize their instantiation as a names dict based on their d.title

    :param sources:
    :return data:
    """
    data = collections.OrderedDict()
    datasets = Parallel(n_jobs=-1)(delayed(DataPackage)(source)
                                   for source in sources)
    for d in datasets:
        data[d.title.tostring()] = d
    return data


def npz_in_dir(path):
    """
    From a given dir, return all the NPZs
    :param path:
    :return sources:
    """
    sources = map(lambda f: os.path.join(os.path.abspath(path), f), filter(
        lambda s: s.endswith('.npz'), os.listdir(path)))
    return sources


def custom_fusion_run(args, data, title):
    # Write intermediate fusion graphs into directories for funzies
    subgraphargs = deepcopy(args)
    subgraphargs.output = 'png'
    subgraphargs.title = title
    subgraphargs.noplot = False
    run_detection_fusion(data=data, args=subgraphargs)


def custom_metric_run(args, data, title):
    # Write intermediate fusion graphs into directories for funzies
    subgraphargs = deepcopy(args)
    subgraphargs.output = 'png'
    subgraphargs.title = title
    subgraphargs.noplot = False
    run_metric_comparison(data=data, args=subgraphargs)


def multirun(args, basedir=os.curdir):
    """
    :param args:
    :return:
    """
    # If doing a multi-run with no given sources, assume we are 1 step away
    # from NPZ's
    if args.source is None:
        sources = filter(
            os.path.isdir, map(os.path.abspath, os.listdir(basedir)))
    args.noplot = True
    get_confidence = lambda x: x['suspect_confidence']
    get_distrust = lambda x: x['suspect_distrust']
    get_name = lambda x: x['suspect_name']
    panel = {}
    best_runs = {}
    for sourcedir in filter(os.path.isdir, sources):
        name = os.path.split(sourcedir)[-1]
        data = load_sources(npz_in_dir(sourcedir))
        execute_generator = lambda d: detect_and_identify(d)
        #execute_generator = lambda d: delayed(detect_and_identify(d))
        result_generator = (execute_generator(d) for d in data.itervalues())
        #result_generator = Parallel(n_jobs=-1)(execute_generator)
        # TODO This is amazingly wasteful
        _, _, _, identification_list = zip(*result_generator)

        # Write a standard fusion run back to the source dir. This might be
        # removed later for speed.
        custom_fusion_run(args, data, sourcedir)

        keys = ['suspect_name', 'suspect_confidence', 'suspect_distrust']
        frame = {}
        for key in keys:
            frame[key] = pd.Series(
                [identification_dict[key]
                    for identification_dict in identification_list]
            )
        frame = pd.DataFrame(frame)
        anonframe = frame.drop('suspect_name', 1)
        filterkey = 'suspect_confidence'
        if "Waypoint" in name:
            best_run_id = anonframe.idxmin(axis=0)[filterkey]
        else:
            best_run_id = anonframe.idxmax(axis=0)[filterkey]

        # Data is an ordered dict so have to flatten it first
        best_runs[name] = deepcopy(data.values()[best_run_id])

        panel[name] = frame
    panel = pd.Panel.from_dict(panel)
    mkpickle("panel.pkl", panel)
    mkpickle("bestruns.pkl", best_runs)
    # Best Worse results plots
    args.strip_title = True
    # annotate achievements in metric, not fusion
    args.annotate_achievements = 1
    plot_runner = custom_metric_run(
        data=best_runs, args=args, title="metric_run")

    # annotate achievements in metric, not fusion
    args.annotate_achievements = 0
    plot_runner = custom_fusion_run(
        data=best_runs, args=args, title="fusion_run")


def custom_parser():
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Simulation Visualisation and Analysis Suite for AIETES",
        epilog="Example Usages:\n"
               "    Plot Metric Values with attempted detection ranges\n"
               "        bounos --shade-region --comparison --attempt-detection --source Stuff.npz\n"
               "    Plot metric fusion (trust fusion with lag-lead)\n"
               "        bounos --fusion --source Stuff.npz\n")
    parser.add_argument('--source', '-s',
                        dest='source', action='store', nargs='+', default=None,
                        metavar='XXX.npz',
                        help='AIETES Simulation Data Package to be analysed')
    parser.add_argument('--output', '-o',
                        dest='output', action='store',
                        default=None,
                        metavar='png|pdf',
                        help='Output to png/pdf')
    parser.add_argument('--title', '-T',
                        dest='title', action='store',
                        default="bounos_figure",
                        metavar='<Filename>',
                        help='Set a title for this analysis run')
    parser.add_argument('--outdim', '-d',
                        dest='dims', action='store', nargs=2, type=int,
                        default=None,
                        metavar='2.3',
                        help='Figure Dimensions in Inches (default autofit)')
    parser.add_argument('--comparison', '-c', dest='compare',
                        action='store_true', default=False,
                        help="Compare Two Datasets for Meta-Analysis")
    parser.add_argument('--fusion', '-f', dest='fusion',
                        action='store_true', default=False,
                        help="Attempt Fusion of Meta-Analysis")
    parser.add_argument('--xkcdify', '-x', dest='xkcd',
                        action='store_true', default=False,
                        help="Plot like Randall")
    parser.add_argument('--analysis', '-a',
                        dest='analysis', action='store', nargs='+', default=None,
                        metavar=str([f[0] for f in list_functions(Analyses)]),
                        help="Select analysis to perform")
    parser.add_argument('--analysis-args', '-A',
                        dest='analysis_args', action='store', default="None",
                        metavar="{'x':1}", type=str,
                        help="Pass on kwargs to analysis in the form of a"
                             "dict to be processed by literaleval")
    parser.add_argument('--no-achievements',
                        dest='achievements', action='store_false', default=True,
                        help="Disable achievement plotting")
    parser.add_argument('--no-plotting',
                        dest='noplot', action='store_true', default=False,
                        help="Disable plotting (only for fusion and identification runs)")
    parser.add_argument('--attempt-detection', '-D',
                        dest='attempt_detection', action='store_true', default=False,
                        help='Attempt Detection and Graphic Annotation for a given analysis')
    parser.add_argument('--shade-region', '-S', dest='shade_region',
                        action='store_true', default=False,
                        help="Shade any detection regions")
    parser.add_argument('--label-size', '-L',
                        dest='font_size', action='store', default=font['size'],
                        metavar=font['size'], type=int,
                        help="Change the Default Font size for axis labels")
    parser.add_argument('--multirun', '-M',
                        dest='multirun', action='store_true', default=False,
                        help="Override normal behaviour to operate with <sources> as directories of simulation runs, to take the best and worst from each, and compare them together (implies fusion and no-noplot along with lots of other hacks)")
    parser.add_argument('--strip-title',
                        dest='strip_title', action='store_true', default=False,
                        help="Strip the title from the last '-', useful for eliminating run-numbers from graphs (only fusion)")
    parser.add_argument('--annotate-achievements', dest='annotate_achievements',
                        type=int, metavar='N',
                        action='store', default=0,
                        help="Annotate the N achievements with info text")
    return parser


def main():
    """
    Initial Entry Point; Does very little other that option parsing
    Raises:
        ValueError if graph selection doesn't make any sense
    """
    global args

    args = custom_parser().parse_args()

    if args.xkcd:
        from XKCDify import XKCDify
    else:
        XKCDify = None

    if args.multirun:
        multirun(args)

    else:
        # if sources is not given, use curdir
        if not args.source:
            sources = npz_in_dir(os.curdir)
        elif isinstance(args.source, list):
            sources = args.source
        else:
            sources = [args.source]

        data = load_sources(sources)

        if args.compare:
            if args.analysis is None:
                # Assuming NxA comparison of sources and analysis (run
                # comparison)
                run_metric_comparison(data, args)
            else:
                raise ValueError(
                    "You're trying to do something stupid: %s" % args)
        elif args.fusion:
            run_detection_fusion(data, args)
        else:
            run_overlay(data, args)


def plot_detections(ax, metric, orig_data,
                    shade_region=False, real_culprits=None, good_behaviour="Waypoint"):
    """
    Plot Detection Overlay including False-positive analysis.

    Will attempt heuristic analysis of 'real' culprit from DataPackage behaviour records

    Args:
        ax(axes): plot to operate on
        metric(Metric): metric to use for detection
        orig_data(DataPackage): data used
        shade_region(bool): shade the detection region (optional:False)
        real_culprits(list): provide a list of culprits for false-positive testing (optional)
        good_behaviour(str): override the default good behaviour (optional: "Waypoint")
    """
    from aietes.Tools import range_grouper

    import Analyses

    results = Analyses.Detect_Misbehaviour(data=orig_data,
                                           metric=metric.__class__.__name__,
                                           stddev_frac=2)
    detections = results['detections']
    detection_vals = results['detection_envelope']
    detection_dict = results['suspicions']

    if real_culprits is None:
        real_culprits = []
    elif isinstance(real_culprits, int):
        real_culprits = [real_culprits]
    else:
        pass

    if good_behaviour:
        for bev, nodelist in orig_data.getBehaviourDict().iteritems():
            if str(good_behaviour) != str(bev):  # Bloody String Comparison...
                print("Adding %s to nodelist because \"%s\" is not \"%s\""
                      % (nodelist, bev, good_behaviour))
                [real_culprits.append(orig_data.names.index(node))
                 for node in nodelist]

    print real_culprits

    for culprit, detections in detection_dict.iteritems():
        for (min, max) in range_grouper(detections):
            if max - min > 20:
                _x = range(min, max)
                if metric.signed is not False:
                    # Negative Detection: Scan from the top
                    _y1 = np.asarray([np.max(metric.data)] * len(_x))
                else:
                    # Positive or Unsigned: Scan from Bottom
                    _y1 = np.asarray([0] * len(_x))
                _y2 = metric.data[min:max, culprit]
                print("%s:%s:%s" %
                      (orig_data.names[culprit], str((min, max)), str(max - min)))
                if real_culprits is not []:
                    ax.fill_between(_x, _y1, _y2, alpha=0.1,
                                    facecolor='red' if culprit not in real_culprits else 'green')
                else:
                    ax.fill_between(_x, _y1, _y2, alpha=0.1, facecolor='red')

    if shade_region:
        _x = np.asarray(range(len(metric.data)))
        ax.fill_between(_x,
                        metric.highlight_data - detection_vals,
                        metric.highlight_data + detection_vals,
                        alpha=0.2, facecolor='red')


def detect_and_identify(d):
    per_metric_deviations, deviation_windowed = Analyses.Combined_Detection_Rank(d,
                                                                                 _metrics,
                                                                                 stddev_frac=2)
    trust_values = Analyses.dev_to_trust(per_metric_deviations)
    identification_dict = Analyses.behaviour_identification(per_metric_deviations, deviation_windowed, _metrics,
                                                            names=d.names,
                                                            verbose=False)
    return trust_values, per_metric_deviations, deviation_windowed, identification_dict


def run_detection_fusion(data, args=None):
    """
    Generate a trust fusion across available metrics, and plot both the metric deviations,
        per-metric detections, and the trust fusion per node, per dataset
    Args:
        data(list of DataPackage): datasets to plot horizontally
        args(argparse.NameSpace): formatting and option arguments (optional)
    """
    import matplotlib.pyplot as plt

    plt.rc('font', **font)
    from matplotlib.pyplot import figure, show, savefig
    from matplotlib.gridspec import GridSpec

    fig = figure()
    base_ax = fig.add_axes([0, 0, 1, 1], )
    gs = GridSpec(len(_metrics) + 1, len(data))
    axes = [[None for _ in range(len(_metrics) + 1)] for _ in range(len(data))]

    # Detect multiple runs by name introspection
    namelist = []
    [namelist.append(name) for (run, d) in data.iteritems()
     for name in d.names]
    nameset = set(namelist)
    per_run_names = len(namelist) / len(data) != len(nameset)

    toptitle = None
    topsuspect = None
    topconfidence = None

    for i, (run, d) in enumerate(sorted(data.items())):
        trust_values, per_metric_deviations, deviation_windowed, identification_dict = detect_and_identify(
            d)
        if args.strip_title and '-' in d.title:
            # Stripping from the last '-' from the title
            d.title = '-'.join(d.title.split('-')[:-1])

        if topconfidence and topconfidence > identification_dict['suspect_confidence']:
            # Not top
            pass
        else:
            toptitle = d.title
            topconfidence = identification_dict['suspect_confidence']
            topsuspect = identification_dict['suspect_name']

        print("{i}:{title} - {suspect}:{confidence:.2f} - Top={toptitle} - {topsuspect}:{topconfidence:.2f}".format(
            i=i, title=d.title, suspect=identification_dict[
                'suspect_name'], confidence=identification_dict['suspect_confidence'],
            toptitle=toptitle, topsuspect=topsuspect, topconfidence=topconfidence)
        )

        if args.noplot:
            continue

        for j, _metric in enumerate(_metrics):
            ax = fig.add_subplot(gs[j, i],
                                 sharex=axes[0][0] if i > 0 or j > 0 else None,
                                 sharey=axes[i - 1][j] if i > 0 else None)
            ax.plot(per_metric_deviations[j], alpha=plot_alpha)

            if args.achievements:
                add_achievements(ax, d)

            ax.grid(True, alpha=0.2)
            ax.autoscale_view(scalex=False, tight=True)
            ax.set_ylabel(_metric.label)
            # First Metric Behaviour (Title)
            if j == 0:
                ax.set_title("{title}\n$\sigma${confidence:.3f}/M-{distrust:.3f}".format(
                             title=str(d.title).replace("_", " "),
                             confidence=identification_dict[
                                 'suspect_confidence'],
                             distrust=identification_dict['suspect_distrust'],
                             )
                             )
                # Last Metric Behaviour (Legend)
            if j == len(_metrics) - 1:
                if per_run_names:
                    # Per Run Legend
                    n_leg = len(data)
                    leg_w = 1.0 / n_leg
                    leg_x = 0 + (leg_w * (i))

                    ax.legend(sorted(d.names), "lower center", bbox_to_anchor=(leg_x, 0, leg_w, 1),
                              bbox_transform=fig.transFigure,
                              ncol=int(ceil(float(len(d.names) + 1) / (n_leg))))
                # First Legend
                elif i == 0:
                    ax.legend(sorted(d.names), "lower center",
                              bbox_to_anchor=(0, 0, 1, 1),
                              bbox_transform=fig.transFigure,
                              ncol=len(d.names))
                else:
                    pass
            else:
                [l.set_visible(False) for l in ax.get_xticklabels()]

            if 'XKCDify' in sys.modules:
                ax = sys.modules.get("XKCDify")(ax)
            axes[i][j] = ax
        j = len(_metrics)
        ax = fig.add_subplot(gs[j, i],
                             sharex=axes[0][0] if i > 0 or j > 0 else None,
                             sharey=axes[i - 1][j] if i > 0 else None)
        ax.grid(True, alpha=0.2)
        mkpickle("trust", trust_values)
        # TODO TRY WITHOUT EWMA
        ax.plot(np.asarray([pd.stats.moments.ewma(t, span=600)
                            for t in trust_values.T]).T, alpha=plot_alpha)
        ax.get_xaxis().set_visible(True)
        ax.set_xlabel("Time ($s$)")
        ax.set_ylabel("Fuzed Trust")
        axes[i][j] = ax

    if args.noplot:

        print("Top={toptitle} - {topsuspect}:{topconfidence:.2f}".format(
            toptitle=toptitle, topsuspect=topsuspect, topconfidence=topconfidence)
        )
        return

    # Now go left to right to adjust the scaling to match

    for j in range(len(axes[0])):
        (m_ymax, m_ymin) = (None, None)
        (m_xmax, m_xmin) = (None, None)
        for i in range(len(axes)):
            (ymin, ymax) = axes[i][j].get_ylim()
            (xmin, xmax) = axes[i][j].get_xlim()
            m_ymax = max(ymax, m_ymax)
            m_ymin = min(ymin, m_ymin)
            m_xmax = max(xmax, m_xmax)
            m_xmin = min(xmin, m_xmin)

        # Do it again to apply the row_max
        for i in range(len(axes)):
            axes[i][j].set_ylim((m_ymin, m_ymax * 1.1))
            axes[i][j].set_xlim((0, m_xmax))

    if args is not None and args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    global_adjust(fig, axes)

    if args is not None and args.output is not None:
        savefig("%s.%s" % (args.title, args.output), bbox_inches=0)
    else:
        show()


def add_achievements(ax, d, annotate_achievements=False):
    if hasattr(d, "achievements"):

        for achievement in d.achievements.nonzero()[1]:
            if annotate_achievements:
                ax.annotate("Vertical lines show\n checkpoint passing points",
                            xy=(achievement, 0.75),
                            xytext=(0.1, 0.8),
                            xycoords='data',
                            textcoords='axes fraction',
                            arrowprops=dict(arrowstyle="->"))
                annotate_achievements = False
            ax.axvline(x=achievement, color='b', alpha=0.1)


def run_metric_comparison(data, args=None):
    """
    Generate available metrics, and plot both the metric values,
        per-metric detections, per dataset
    Args:
        data(list of DataPackage): datasets to plot horizontally
        args(argparse.NameSpace): formatting and option arguments (optional)
    """
    import matplotlib.pyplot as plt

    plt.rc('font', **font)
    from matplotlib.pyplot import figure, show, savefig
    from matplotlib.gridspec import GridSpec

    fig = figure()
    base_ax = fig.add_axes([0, 0, 1, 1], )
    gs = GridSpec(len(_metrics), len(data))

    # Detect multiple runs by name introspection
    namelist = []
    [namelist.append(name) for (run, d) in data.iteritems()
     for name in d.names]
    nameset = set(namelist)
    per_run_names = len(namelist) / len(data) != len(nameset)

    axes = [[None for _ in range(len(_metrics))] for _ in range(len(data))]
    for i, (run, d) in enumerate(sorted(data.iteritems())):
        if args.strip_title and '-' in d.title:
            # Stripping from the last '-' from the title
            d.title = '-'.join(d.title.split('-')[:-1])
        for j, _metric in enumerate(_metrics):
            metric = _metric(data=d)
            metric.update()
            # Sharing axis is awkward; each metric should be matched to it's partners,
            # and x axis should always be shared
            if i > 0 or j > 0:
                # This is the axis that will be shared for time
                sharedx = axes[0][0]
            else:
                sharedx = None
            if i > 0:
                # Each 'row' shares a y axis
                sharedy = axes[i - 1][j]
            else:
                sharedy = None
            ax = fig.add_subplot(gs[j, i], sharex=sharedx, sharey=sharedy)
            ax.plot(metric.data, alpha=plot_alpha)
            if metric.highlight_data is not None:
                ax.plot(
                    metric.highlight_data, color='k', linestyle='--', alpha=plot_alpha)

            if args.achievements:
                if args.annotate_achievements >= 1:
                    add_achievements(
                        ax, d, annotate_achievements=args.annotate_achievements)
                    args.annotate_achievements -= 1
                else:
                    add_achievements(
                        ax, d, annotate_achievements=args.annotate_achievements)

            if args is not None and args.attempt_detection:
                plot_detections(ax, metric, d, shade_region=args.shade_region)

            ax.grid(True, alpha=0.2)
            ax.autoscale_view(scalex=False, tight=True)
            ax.set_ylabel(_metric.label)
            # First Metric Behaviour (Title)
            if j == 0:
                ax.set_title(str(d.title).replace("_", " "))
                # Last Metric Behaviour (Legend)
            if j == len(_metrics) - 1:
                if per_run_names:
                    # Per Run Legend
                    n_leg = len(data)
                    leg_w = 1.0 / n_leg
                    leg_x = 0 + (leg_w * (i))

                    ax.legend(d.names, "lower center",
                              bbox_to_anchor=(leg_x, 0, leg_w, 1),
                              bbox_transform=fig.transFigure,
                              ncol=int(ceil(float(len(d.names) + 1) / (n_leg))))
                # First Legend
                elif i == 0:
                    ax.legend(d.names, "lower center",
                              bbox_to_anchor=(0, 0, 1, 1),
                              bbox_transform=fig.transFigure,
                              ncol=len(d.names))
            # Turn off ALL metric xticks
            [l.set_visible(False) for l in ax.get_xticklabels()]
            axes[i][j] = ax

    # Now go left to right to adjust the scaling to match
    for j in range(len(axes[0])):
        (m_ymax, m_ymin) = (None, None)
        (m_xmax, m_xmin) = (None, None)
        for i in range(len(axes)):
            (ymin, ymax) = axes[i][j].get_ylim()
            (xmin, xmax) = axes[i][j].get_xlim()
            m_ymax = max(ymax, m_ymax)
            m_ymin = min(ymin, m_ymin)
            m_xmax = max(xmax, m_xmax)
            m_xmin = min(xmin, m_xmin)

        # Do it again to apply the row_max
        for i in range(len(axes)):
            axes[i][j].set_ylim((m_ymin, m_ymax * 1.1))
            axes[i][j].set_xlim((0, m_xmax))

    if args is not None and args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    global_adjust(fig, axes)

    if args is not None and args.output is not None:
        savefig("%s.%s" % (args.title, args.output), bbox_inches=0)
    else:
        show()


def run_overlay(data, args=None):
    # TODO Documentation
    """
    Args:
        data(list of DataPackage): datasets to plot horizontally
        args(argparse.NameSpace): formatting and option arguments (optional)
    """
    import pylab as pl
    from ast import literal_eval

    pl.rc('font', **font)

    analysis = getattr(Analyses, args.analysis[0])
    try:
        analysis_args = literal_eval(args.analysis_args)
    except ValueError as exp:
        print args.analysis_args
        raise exp
    finally:
        print analysis_args

    fig = pl.figure()
    ax = fig.gca()

    results = {}
    for source in data.keys():
        if analysis_args is not None:
            results = analysis(data=data[source], **analysis_args)
        else:
            results = analysis(data=data[source])
        metrics = results['detection_envelope']
        if hasattr(results, 'detections'):
            detections = results['detections']
        elif args.attempt_detection:
            raise NotImplementedError(
                "Tried to do detection on a metric that doesn't support it:%s" % args.analysis)
        else:
            pass

        ax.plot(metrics, label=str(data[source].title).replace(
            "_", " "), alpha=plot_alpha)
        try:
            if args.attempt_detection:
                ax.fill_between(range(len(metrics)),
                                0, metrics,
                                where=[d is not None for d in detections], alpha=0.3)
            else:
                ax.fill_between(range(len(metrics)),
                                metrics -
                                data[source].data, metrics + data[source].data,
                                alpha=0.2, facecolor='red')
        except ValueError as exp:
            print("Metrics:%s" % str(metrics.shape))
            print("Data:%s" % str(data[source]))
            print("Detections:%s" % str(detections.shape))
            raise exp

    ax.legend(loc="upper right", prop={'size': 12})
    ax.set_title(str(args.title).replace("_", " "))
    ax.set_ylabel(analysis.__name__.replace("_", " "))
    ax.set_xlabel("Time ($s$)")

    global_adjust(fig, ax)

    if args.output is not None:
        pl.savefig("%s.%s" % (args.title, args.output), bbox_inches=0)
    else:
        pl.show()


def global_adjust(figure, axes, scale=2):
    """
    General Figure adjustments:
        Subplot-spacing adjustments
        Figure sizing/scaling
    Args:
        figure(Figure): figure to be adjusted
        scale(int/float): adjust figure to scale (optional:2)
    """
    def math_formatter(x, pos):
        return "$%s$" % x

    for axe in axes:
        for ax in axe:
            if 'args' in locals():
                ax.set_ylabel(ax.get_ylabel(), size=args.font_size)
            if 'XKCDify' in sys.modules:
                ax = sys.modules.get("XKCDify")(ax)
            ax.yaxis.set_major_formatter(FuncFormatter(math_formatter))

    figure.set_size_inches(figure.get_size_inches() * scale)
    figure.subplots_adjust(
        left=0.05, bottom=0.1, right=0.98, top=0.95, wspace=0.2, hspace=0.0)


def printAnalysis(d):
    deviation, trust = Analyses.Combined_Detection_Rank(
        d, _metrics, stddev_frac=2)
    result_dict = Analyses.behaviour_identification(
        deviation, trust, _metrics, names=d.names)
    return result_dict

if __name__ == '__main__':
    main()
