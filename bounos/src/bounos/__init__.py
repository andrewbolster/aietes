# BOUNOS - Heir to the Kingdom of AIETES

import sys
import os
import traceback
import argparse

import numpy as np
import scipy as sp
import matplotlib
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import mpl_toolkits.mplot3d.axes3d as axes3
from matplotlib.widgets import Slider, Button, RadioButtons

class DataPackage():
    """
    Data Store for simulation results
    Replicates a Numpy Array as is directly accessible as an n-vector
    [x,y,z,t] array for n simulated nodes positions over time. Additional
    information is queriable through the object.
    """

    def __init__(self,source,*args,**kwargs):
        #TODO Try Except this to be safe
        source_dataset = np.load(source)
        self.p = source_dataset['positions']
        self.v = source_dataset['vectors']
        self.names = source_dataset['names']
        self.environment = source_dataset['environment']
        self.tmax = len(self.p[0][0])
        self.n = len(self.p)
        try:
            self.title = getattr(source_dataset,'title')
        except AttributeError:
            # If simulation title not explicitly given, use the filename -npz
            self.title = os.path.splitext(os.path.basename(source))[0]

        #Data has the format:
            # [n][x,y,z][t]

    def position_of(self,node,time):
        """
        Query the data set for the x,y,z position of a node at a given time
        """
        return [self.p[node][dimension][time] for dimension in 0,1,2]

    def heading_of(self,node,time):
        """
        Query the data set for the x,y,z vector of a node at a given time
        """
        return [self.v[node][dimension][time] for dimension in 0,1,2]

    def heading_slice(self,time):
        """
        Query the dataset for the [n][x,y,z] heading list of all nodes at a given time
        """
        return [ self.heading_of(x,time) for x in range(self.n) ]


    def trail_of(self,node,time=None):
        """
        Return the [X:][Y:][Z:] trail for a given node from sim_start to 
        a particular time

        If no time given, assume the full time range
        """
        if time is None:
            time = self.tmax

        return [self.p[node][dimension][0:time] for dimension in 0,1,2]

def main():
    """
    Initial Entry Point; Does very little other that option parsing
    """
    parser = argparse.ArgumentParser(description="Simulation Visualisation and Analysis Suite for AIETES")
    parser.add_argument('--source','-s',
                        dest='source', action='store',
                        metavar='XXX.npz',
                        required=True,
                        help='AIETES Simulation Data Package to be analysed'
                       )


    args=parser.parse_args()
    interactive_plot(DataPackage(args.source))

def interactive_plot(data):
    """
    Generate the MPL data browser for the flock data
    """
    # Generate Arrangements for viewport + accessory views
    plt.close('all')
    fig = plt.figure()
    gs = GridSpec(3,4)
    ax = plt.subplot(gs[:,1:], projection='3d')
    axH = plt.subplot(gs[:,0])

    # Find initial display state for viewport
    lines = [ ax.plot( xs, ys, zs)[0] for xs,ys,zs in data.p ]
    for n,line in enumerate(lines):
        line.set_label(data.names[n])

    #Configure the Time Slider
    timeax = plt.axes([0.2, 0.0, 0.65, 0.03])
    timeslider = Slider(timeax, 'Time', 0, data.tmax, valinit=0)


    # Set initial Vector Display
    vectors = data.heading_slice(0)
    count = len(vectors)
    ind = np.arange(count)
    width = 0.11

    global rectX,rectY,rectZ
    rectX=axH.barh(ind, tuple([vec[0] for vec in vectors]), width, color='r')
    rectY=axH.barh(ind+width, tuple([vec[1] for vec in vectors]), width, color='g')
    rectZ=axH.barh(ind+2*width, tuple([vec[2] for vec in vectors]), width, color='b')


    def press(event):
        if event.key is 'left':
            timeslider.set_val(timeslider.val-1)
        elif event.key is 'right':
            timeslider.set_val(timeslider.val+1)
        else:
            print('press', event.key)


    fig.canvas.mpl_connect('key_press_event', press)

    def update_viewport(val):
        """
        Update Line display across time
        """
        for n,line in enumerate(lines):
            (xs,ys,zs)=data.trail_of(n,timeslider.val)
            line.set_data(xs,ys)
            line.set_3d_properties(zs)
            line.set_label(data.names[n])

    def update_headings(val):
        """
        Update Vector Heading display across time
        """
        vectors = data.heading_slice(timeslider.val)
        axH.cla()
        rectX=axH.barh(ind, tuple([vec[0] for vec in vectors]), width, color='r')
        rectY=axH.barh(ind+width, tuple([vec[1] for vec in vectors]), width, color='g')
        rectZ=axH.barh(ind+2*width, tuple([vec[2] for vec in vectors]), width, color='b')

        axH.set_ylabel("Vector")
        axH.set_yticks(ind+width)
        axH.set_yticklabels( (data.names) )

    def update(val):
        if timeslider.val > data.tmax:
            timeslider.set_val(data.tmax)
            return
        if timeslider.val < 0:
            timeslider.set_val(0)
            return

        update_viewport(val)
        update_headings(val)
        plt.draw()

    timeslider.on_changed(update)

    shape = data.environment
    ax.legend()
    ax.set_title("Tracking overview of %s"%data.title)
    ax.set_xlim3d((0,shape[0]))
    ax.set_ylim3d((0,shape[1]))
    ax.set_zlim3d((0,shape[2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    plt.show()

