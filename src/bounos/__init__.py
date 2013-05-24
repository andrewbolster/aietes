"""
BOUNOS - Heir to the Kingdom of AIETES
"""

import sys
import re
import argparse
from argparse import RawTextHelpFormatter

from math import ceil

import numpy as np


np.seterr(under="ignore")

from Metrics import *
import Analyses
from DataPackage import DataPackage

from aietes.Tools import list_functions

font = {'family': 'normal',
        'weight': 'normal',
        'size': 10}
_metrics = [Deviation_Of_Heading,
            PerNode_Speed,
            PerNode_Internode_Distance_Avg]


class BounosModel(DataPackage):
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
        Will 'export' DataPackage data from the running simulation up to the requested time (self.t)
        """
        self.log.debug("Updating data from simulator at %d" % now)
        self.update(p=p, v=v, names=names, environment=environment)

    @classmethod
    def is_valid_aietes_datafile(cls, file):
        #TODO This isn't a very good test...
        test = re.compile(".npz$")
        return test.search(file)


def main():
    """
    Initial Entry Point; Does very little other that option parsing
    """
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Simulation Visualisation and Analysis Suite for AIETES",
        epilog="Example Usages:\n" \
               "    Plot Metric Values with attempted detection ranges\n" \
               "        bounos --shade-region --comparison --attempt-detection --source Stuff.npz\n" \
               "    Plot metric fusion (trust fusion with lag-lead)\n" \
               "        bounos --fusion --source Stuff.npz\n")
    parser.add_argument('--source', '-s',
                        dest='source', action='store', nargs='+',
                        metavar='XXX.npz',
                        required=True,
                        help='AIETES Simulation Data Package to be analysed'
    )
    parser.add_argument('--output', '-o',
                        dest='output', action='store',
                        default=None,
                        metavar='png|pdf',
                        help='Output to png/pdf'
    )
    parser.add_argument('--title', '-T',
                        dest='title', action='store',
                        default="bounos_figure",
                        metavar='<Filename>',
                        help='Set a title for this analysis run'
    )
    parser.add_argument('--outdim', '-d',
                        dest='dims', action='store', nargs=2, type=int,
                        default=None,
                        metavar='2.3',
                        help='Figure Dimensions in Inches (default autofit)'
    )
    parser.add_argument('--comparison', '-c', dest='compare',
                        action='store_true', default=False,
                        help="Compare Two Datasets for Meta-Analysis"
    )
    parser.add_argument('--fusion', '-f', dest='fusion',
                        action='store_true', default=False,
                        help="Attempt Fusion of Meta-Analysis"
    )
    parser.add_argument('--xkcdify', '-x', dest='xkcd',
                        action='store_true', default=False,
                        help="Plot like Randall"
    )
    parser.add_argument('--analysis', '-a',
                        dest='analysis', action='store', nargs='+', default=None,
                        metavar=str([f[0] for f in list_functions(Analyses)]),
                        help="Select analysis to perform"
    )
    parser.add_argument('--analysis-args', '-A',
                        dest='analysis_args', action='store', default="None",
                        metavar="{'x':1}", type=str,
                        help="Pass on kwargs to analysis in the form of a dict to be processed by literaleval"
    )

    parser.add_argument('--attempt-detection', '-D',
                        dest='attempt_detection', action='store_true', default=False,
                        help='Attempt Detection and Graphic Annotation for a given analysis'
    )
    parser.add_argument('--shade-region', '-S', dest='shade_region',
                        action='store_true', default=False,
                        help="Shade any detection regions"
    )

    args = parser.parse_args()
    print args

    if isinstance(args.source, list):
        sources = args.source
    else:
        sources = [args.source]

    data = {}
    for source in sources:
        data[source] = DataPackage(source)

    if args.xkcd:
        from XKCDify import XKCDify

    if args.compare:
        if args.analysis is None:
            #Assuming NxA comparison of sources and analysis (run comparison)
            run_metric_comparison(data, args)
        else:
            raise ValueError("You're trying to do something stupid: %s" % args)
    elif args.fusion:
        run_detection_fusion(data, args)
    else:
        run_overlay(data, args)


def plot_detections(ax, metric, orig_data, shade_region=False, real_culprits=None, good_behaviour="Waypoint"):
    """
    Plot Detection Overlay including False-positive analysis.

    Will attempt heuristic analysis of 'real' culprit from DataPackage behaviour records

    Keyword Arguments:
        shade_region:bool(False)
        real_culprits:list([])
        good_behaviour:str("Waypoint")
    """
    from aietes.Tools import range_grouper

    import Analyses

    results = Analyses.Detect_Misbehaviour(data=orig_data,
                                           metric=metric.__class__.__name__,
                                           stddev_frac=2)
    (detections, detection_vals, detection_dict) = results['detections'], results['detection_envelope'], results[
        'suspicions']
    if real_culprits is None:
        real_culprits = []
    elif isinstance(real_culprits, int):
        real_culprits = [real_culprits]
    else:
        pass

    if good_behaviour:
        for bev, nodelist in orig_data.getBehaviourDict().iteritems():
            if str(good_behaviour) != str(bev):  # Bloody String Comparison...
                print "Adding %s to nodelist because \"%s\" is not \"%s\"" % (nodelist, bev, good_behaviour)
                [real_culprits.append(orig_data.names.index(node)) for node in nodelist]

    print real_culprits

    for culprit, detections in detection_dict.iteritems():
        for (min, max) in range_grouper(detections):
            if max - min > 20:
                _x = range(min, max)
                if metric.signed is not False:
                    #Negative Detection: Scan from the top
                    _y1 = np.asarray([np.max(metric.data)] * len(_x))
                else:
                    # Positive or Unsigned: Scan from Bottom
                    _y1 = np.asarray([0] * len(_x))
                _y2 = metric.data[min:max, culprit]
                print("%s:%s:%s" % (orig_data.names[culprit], str((min, max)), str(max - min)))
                if real_culprits is not []:
                    ax.fill_between(_x, _y1, _y2, alpha=0.1,
                                    facecolor='red' if culprit not in real_culprits else 'green')
                else:
                    ax.fill_between(_x, _y1, _y2, alpha=0.1, facecolor='red')

    if shade_region:
        _x = np.asarray(range(len(metric.data)))
        ax.fill_between(_x, metric.highlight_data - detection_vals, metric.highlight_data + detection_vals, alpha=0.2,
                        facecolor='red')


def run_detection_fusion(data, args):
    import matplotlib.pyplot as plt

    plt.rc('font', **font)
    from matplotlib.pyplot import figure, show, savefig
    from matplotlib.gridspec import GridSpec
    import Metrics


    fig = figure()
    base_ax = fig.add_axes([0, 0, 1, 1], )
    gs = GridSpec(len(_metrics) + 1, len(data))
    axes = [[None for _ in range(len(_metrics) + 1)] for _ in range(len(data))]

    #Detect multiple runs by name introspection
    namelist = []
    [namelist.append(name) for (run, d) in data.iteritems() for name in d.names]
    nameset = set(namelist)
    per_run_names = len(namelist) / len(data) != len(nameset)

    for i, (run, d) in enumerate(data.iteritems()):
        print("One: %d" % i)
        deviation_fusion, deviation_windowed = Analyses.Combined_Detection_Rank(d, _metrics, stddev_frac=2)
        for j, _metric in enumerate(_metrics):
            print("One: %d:%d" % (i, j))

            ax = fig.add_subplot(gs[j, i],
                                 sharex=axes[0][0] if i > 0 or j > 0 else None,
                                 sharey=axes[i - 1][j] if i > 0 else None)
            ax.plot(deviation_fusion[j])

            if hasattr(d, "achievements"):
                for achievement in d.achievements.nonzero()[1]:
                    ax.axvline(x=achievement, color='b', alpha=0.1)

            ax.grid(True, alpha='0.2')
            ax.autoscale_view(scalex=False, tight=True)
            # First Dataset B1haviour
            if i == 0:
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

                    ax.legend(d.names, "lower center", bbox_to_anchor=(leg_x, 0, leg_w, 1),
                              bbox_transform=fig.transFigure,
                              ncol=int(ceil(float(len(d.names)) / (n_leg))))
                #First Legend
                elif i == 0:
                    ax.legend(d.names, "lower center", bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure,
                              ncol=len(d.names))
                else:
                    pass
            else:
                [l.set_visible(False) for l in ax.get_xticklabels()]
            if 'XKCDify' in sys.modules:
                ax = XKCDify(ax)
            axes[i][j] = ax
        j = len(_metrics)
        ax = fig.add_subplot(gs[j, i],
                             sharex=axes[0][0] if i > 0 or j > 0 else None,
                             sharey=axes[i - 1][j] if i > 0 else None)
        ax.plot(deviation_windowed)
        ax.get_xaxis().set_visible(True)
        ax.set_xlabel("Time")
        ax.set_ylabel("Fuzed Trust")
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

        #Do it again to apply the row_max
        for i in range(len(axes)):
            axes[i][j].set_ylim((m_ymin, m_ymax * 1.1))
            axes[i][j].set_xlim((0, m_xmax))

    if args is not None and args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    resize(fig, 2)

    if args is not None and args.output is not None:
        savefig("%s.%s" % (args.title, args.output), bbox_inches=0)
    else:
        show()


def run_metric_comparison(data, args):
    import matplotlib.pyplot as plt

    plt.rc('font', **font)
    from matplotlib.pyplot import figure, show, savefig
    from matplotlib.gridspec import GridSpec

    fig = figure()
    base_ax = fig.add_axes([0, 0, 1, 1], )
    gs = GridSpec(len(_metrics), len(data))

    #Detect multiple runs by name introspection
    namelist = []
    [namelist.append(name) for (run, d) in data.iteritems() for name in d.names]
    nameset = set(namelist)
    per_run_names = len(namelist) / len(data) != len(nameset)

    axes = [[None for _ in range(len(_metrics))] for _ in range(len(data))]
    for i, (run, d) in enumerate(data.iteritems()):
        for j, _metric in enumerate(_metrics):
            metric = _metric(data=d)
            metric.update()
            #Sharing axis is awkward; each metric should be matched to it's partners, and x axis should always be shared
            if i > 0 or j > 0:
                #This is the axis that will be shared for time
                sharedx = axes[0][0]
            else:
                sharedx = None
            if i > 0:
                #Each 'row' shares a y axis
                sharedy = axes[i - 1][j]
            else:
                sharedy = None
            ax = fig.add_subplot(gs[j, i], sharex=sharedx, sharey=sharedy)
            ax.plot(metric.data, alpha=0.3)
            if metric.highlight_data is not None:
                ax.plot(metric.highlight_data, color='k', linestyle='--')

            if hasattr(d, "achievements"):
                for achievement in d.achievements.nonzero()[1]:
                    ax.axvline(x=achievement, color='b', alpha=0.1)

            #if args.attempt_detection and isinstance(metric, Metrics.PerNode_Internode_Distance_Avg):
            if args is not None and args.attempt_detection:
                plot_detections(ax, metric, d, shade_region=args.shade_region)

            ax.grid(True, alpha='0.2')
            ax.autoscale_view(scalex=False, tight=True)
            # First Dataset Behaviour
            if i == 0:
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

                    ax.legend(d.names, "lower center", bbox_to_anchor=(leg_x, 0, leg_w, 1),
                              bbox_transform=fig.transFigure,
                              ncol=int(ceil(float(len(d.names)) / (n_leg))))
                #First Legend
                elif i == 0:
                    ax.legend(d.names, "lower center", bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure,
                              ncol=len(d.names))
            else:
                [l.set_visible(False) for l in ax.get_xticklabels()]
            if 'XKCDify' in sys.modules:
                ax = XKCDify(ax)
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

        #Do it again to apply the row_max
        for i in range(len(axes)):
            axes[i][j].set_ylim((m_ymin, m_ymax * 1.1))
            axes[i][j].set_xlim((0, m_xmax))

    if args is not None and args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    resize(fig, 2)

    if args is not None and args.output is not None:
        savefig("%s.%s" % (args.title, args.output), bbox_inches=0)
    else:
        show()


def run_overlay(data, args):
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
            raise NotImplementedError("Tried to do detection on a metric that doesn't support it:%s" % args.analysis)
        else:
            pass

        ax.plot(metrics, label=str(data[source].title).replace("_", " "))
        try:
            if args.attempt_detection:
                ax.fill_between(range(len(metrics)), 0, metrics, where=[d is not None for d in detections], alpha=0.3)
            else:
                ax.fill_between(range(len(metrics)), metrics - data[source].data, metrics + data[source].data,
                                alpha=0.2, facecolor='red')
        except ValueError as exp:
            print("Metrics:%s" % str(metrics.shape))
            print("Data:%s" % str(data[source]))
            print("Detections:%s" % str(detections.shape))
            raise exp

    ax.legend(loc="upper right", prop={'size': 12})
    ax.set_title(str(args.title).replace("_", " "))
    ax.set_ylabel(analysis.__name__.replace("_", " "))
    ax.set_xlabel("Time")

    if 'XKCDify' in sys.modules:
        ax = XKCDify(ax)

    if args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    if args.output is not None:
        pl.savefig("%s.%s" % (args.title, args.output), bbox_inches=0)
    else:
        pl.show()


def resize(figure, scale):
    figure.set_size_inches(figure.get_size_inches() * scale)
    figure.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.95, wspace=0.2, hspace=0.0)
