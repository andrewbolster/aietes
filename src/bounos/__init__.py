# BOUNOS - Heir to the Kingdom of AIETES

import re

import argparse
import numpy as np

from Plotting import *

np.seterr(under = "ignore")

from Metrics import *
import Analyses
from DataPackage import DataPackage
from XKCDify import XKCDify

from aietes.Tools import list_functions

font = {'family': 'normal',
        'weight': 'normal',
        'size': 10}


class BounosModel(DataPackage):
    def __init__(self, *args, **kwargs):
        self.metrics = []
        self.is_ready = False
        self.is_simulating = None

    def import_datafile(self, file):
        super(BounosModel, self).__init__(source = file)
        self.is_ready = True
        self.is_simulating = False


    def update_data_from_sim(self, p, v, names, environment, now):
        """
        Call back function used by SimulationStep if doing real time simulation
        Will 'export' DataPackage data from the running simulation up to the requested time (self.t)
        """
        self.log.debug("Updating data from simulator at %d" % now)
        self.update(p = p, v = v, names = names, environment = environment)

    @classmethod
    def is_valid_aietes_datafile(cls, file):
        #TODO This isn't a very good test...
        test = re.compile(".npz$")
        return test.search(file)


def main():
    """
    Initial Entry Point; Does very little other that option parsing
    """
    parser = argparse.ArgumentParser(description = "Simulation Visualisation and Analysis Suite for AIETES")
    parser.add_argument('--source', '-s',
                        dest = 'source', action = 'store', nargs = '+',
                        metavar = 'XXX.npz',
                        required = True,
                        help = 'AIETES Simulation Data Package to be analysed'
    )
    parser.add_argument('--output', '-o',
                        dest = 'output', action = 'store',
                        default = None,
                        metavar = 'png|pdf',
                        help = 'Output to png/pdf'
    )
    parser.add_argument('--title', '-T',
                        dest = 'title', action = 'store',
                        default = "bounos_figure",
                        metavar = '<Filename>',
                        help = 'Set a title for this analysis run'
    )
    parser.add_argument('--outdim', '-d',
                        dest = 'dims', action = 'store', nargs = 2, type = int,
                        default = None,
                        metavar = '2.3',
                        help = 'Figure Dimensions in Inches (default autofit)'
    )
    parser.add_argument('--comparison', '-c', dest = 'compare',
                        action = 'store_true', default = False,
                        help = "Compare Two Datasets for Meta-Analysis"
    )
    parser.add_argument('--fusion', '-f', dest = 'fusion',
                        action = 'store_true', default = False,
                        help = "Attempt Fusion of Meta-Analysis"
    )
    parser.add_argument('--xkcdify', '-x', dest = 'xkcd',
                        action = 'store_true', default = False,
                        help = "Plot like Randall"
    )
    parser.add_argument('--analysis', '-a',
                        dest = 'analysis', action = 'store', nargs = '+', default = None,
                        metavar = str([f[0] for f in list_functions(Analyses)]),
                        help = "Select analysis to perform"
    )
    parser.add_argument('--analysis-args', '-A',
                        dest = 'analysis_args', action = 'store', default = None,
                        metavar = "{'x':1}", type= str,
                        help = "Pass on kwargs to analysis in the form of a dict to be processed by literaleval"
    )

    parser.add_argument('--attempt-detection', '-D',
                        dest = 'attempt_detection', action = 'store_true', default = False,
                        help = 'Attempt Detection and Graphic Annotation for a given analysis'
    )
    parser.add_argument('--shade-region', '-S', dest = 'shade_region',
                        action = 'store_true', default = False,
                        help = "Shade any detection regions"
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

def plot_detections(ax, metric, orig_data, shade_region=False, real_culprit=None):
    from aietes.Tools import range_grouper

    import Analyses
    (detections, detection_vals, detection_dict, _) = Analyses.Detect_Misbehaviour(data = orig_data, metric=metric.__class__.__name__)

    for culprit, detections in detection_dict.iteritems():
        for (min,max) in range_grouper(detections):
            if max-min > 20:
                _x = range(min,max)
                if metric.signed is not False:
                    #Negative Detection: Scan from the top
                    _y1 = np.asarray([np.max(metric.data)]*len(_x))
                else:
                    # Positive or Unsigned: Scan from Bottom
                    _y1 = np.asarray([0]*len(_x))
                _y2 = metric.data[min:max,culprit]
                print("%s:%s:%s"%(orig_data.names[culprit], str((min,max)), str(max-min)))
                if real_culprit is not None:
                    ax.fill_between(_x, _y1, _y2 , alpha=0.1, facecolor='red' if culprit != real_culprit else 'green')
                else:
                    ax.fill_between(_x, _y1, _y2 , alpha=0.1, facecolor='red')



    if shade_region:
        _x = np.asarray(range(len(metric.data)))
        ax.fill_between(_x, metric.highlight_data - detection_vals, metric.highlight_data+detection_vals, alpha=0.2, facecolor='red' )

def run_detection_fusion(data, args):
    import matplotlib.pyplot as plt

    plt.rc('font', **font)
    from matplotlib.pyplot import figure, show, savefig
    from matplotlib.gridspec import GridSpec
    import Metrics

    _metrics = [Metrics.Deviation_Of_Heading,
                Metrics.PerNode_Speed,
                Metrics.PerNode_Internode_Distance_Avg]

    fig = figure()
    base_ax = fig.add_axes([0, 0, 1, 1], )
    gs = GridSpec(len(_metrics)+1, len(data))


    axes =[[None for _ in range(len(_metrics) + 1)] for _ in range(len(data))]

    for i, (run, d) in enumerate(data.iteritems()):
        print("One: %d"%i)
        deviation_fusion, deviation_windowed = Analyses.Combined_Detection_Rank(d, _metrics)
        for j, _metric in enumerate(_metrics):
            print("One: %d:%d"%(i,j))
            ax = fig.add_subplot(gs[j, i])
            ax.plot(deviation_fusion[j])

            ax.grid(True, alpha = '0.2')
            ax.autoscale_view(scalex = False, tight = True)
            # First Dataset Behaviour
            if i == 0:
                ax.set_ylabel(_metric.label)
                # First Metric Behaviour (Title)
            if j == 0:
                ax.set_title(d.title.replace("_", " "))
                # Last Metric Behaviour (Legend)
            if j == len(_metrics) - 1:
                if i == 0:
                    ax.legend(d.names, "lower center", bbox_to_anchor = (0, 0, 1, 1), bbox_transform = fig.transFigure,
                              ncol = len(d.names))
            else:
                [l.set_visible(False) for l in ax.get_xticklabels()]
            if args.xkcd:
                ax = XKCDify(ax)
            axes[i][j] = ax
        j = len(_metrics)
        ax = fig.add_subplot(gs[j, i])
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

    if args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    resize(fig, 2)

    if args.output is not None:
        savefig("%s.%s" % (args.title, args.output), bbox_inches = 0)
    else:
        show()



def run_metric_comparison(data, args):
    import matplotlib.pyplot as plt

    plt.rc('font', **font)
    from matplotlib.pyplot import figure, show, savefig
    from matplotlib.gridspec import GridSpec
    import Metrics

    _metrics = [Metrics.Deviation_Of_Heading,
                Metrics.PerNode_Speed,
                Metrics.PerNode_Internode_Distance_Avg]

    fig = figure()
    base_ax = fig.add_axes([0, 0, 1, 1], )
    gs = GridSpec(len(_metrics), len(data))

    axes = [[None for _ in range(len(_metrics))] for _ in range(len(data))]
    for i, (run, d) in enumerate(data.iteritems()):
        for j, _metric in enumerate(_metrics):
            metric = _metric(data = d)
            metric.update()
            ax = fig.add_subplot(gs[j, i])
            ax.plot(metric.data, alpha = 0.3)
            if metric.highlight_data is not None:
                ax.plot(metric.highlight_data, color = 'k', linestyle = '--')

            #if args.attempt_detection and isinstance(metric, Metrics.PerNode_Internode_Distance_Avg):
            if args.attempt_detection:
                plot_detections(ax, metric, d, shade_region=args.shade_region, real_culprit=1 if i==0 else None)

            ax.grid(True, alpha = '0.2')
            ax.autoscale_view(scalex = False, tight = True)
            # First Dataset Behaviour
            if i == 0:
                ax.set_ylabel(_metric.label)
            # First Metric Behaviour (Title)
            if j == 0:
                ax.set_title(d.title.replace("_", " "))
            # Last Meric Behaviour (Legend)
            if j == len(_metrics) - 1:
                ax.get_xaxis().set_visible(True)
                ax.set_xlabel("Time")
                if i == 0:
                    ax.legend(d.names, "lower center", bbox_to_anchor = (0, 0, 1, 1), bbox_transform = fig.transFigure,
                              ncol = len(d.names))
            else:
                [l.set_visible(False) for l in ax.get_xticklabels()]
            if args.xkcd:
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

    if args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    resize(fig, 2)

    if args.output is not None:
        savefig("%s.%s" % (args.title, args.output), bbox_inches = 0)
    else:
        show()


def run_overlay(data, args):
    import pylab as pl
    from ast import literal_eval

    pl.rc('font', **font)

    analysis = getattr(Analyses, args.analysis[0])
    analysis_args = literal_eval(args.analysis_args)
    print analysis_args

    fig = pl.figure()
    ax = fig.gca()

    results = {}
    for source in data.keys():
        #interactive_plot(data)
        (detections, metrics) = analysis(data = data[source], **analysis_args)
        ax.plot(metrics, label = data[source].title.replace("_", " "))
        if args.attempt_detection:
            ax.fill_between(range(len(metrics)), 0, metrics, where=[ d is not None for d in detections], alpha = 0.3)
        else:
            ax.fill_between(range(len(metrics)), metrics - data[source].data, metrics + data[source].data, alpha=0.2, facecolor='red' )


    ax.legend(loc = "upper right", prop = {'size': 12})
    ax.set_title(args.title.replace("_", " "))
    ax.set_ylabel(analysis.__name__.replace("_", " "))
    ax.set_xlabel("Time")

    if args.xkcd:
        ax = XKCDify(ax)

    if args.dims is not None:
        fig.set_size_inches((int(d) for d in args.dims))

    if args.output is not None:
        pl.savefig("%s.%s" % (args.title, args.output), bbox_inches = 0)
    else:
        pl.show()


def resize(figure, scale):
    figure.set_size_inches(figure.get_size_inches() * scale)
