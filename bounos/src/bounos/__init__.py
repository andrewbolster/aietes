# Example package with a console entry point

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3
from matplotlib.widgets import Slider, Button, RadioButtons

class DataPackage():
    """
    Data Store for simulation results
    Replicates a Numpy Array as is directly accessible as an n-vector
    [x,y,z,t] array for n simulated nodes positions over time. Additional
    information is queriable through the object.
    """

    def __init__(self,*args,**kwargs):
        #TODO Try Except this to be safe
        source_dataset = np.load(kwargs.get('source'))
        self.d = source_dataset['data']
        self.names = source_dataset['names']
        self.environment = source_dataset['environment']
        self.tmax = len(self.d[0][0])

        #Data has the format:
            # [n][x,y,z][t]

    def position_of(self,node,time):
        """
        Query the data set for the x,y,z position of a node at a given time
        """
        return [self.d[node][dimension][time] for dimension in 0,1,2]

    def trail_of(self,node,time=None):
        """
        Return the [X:][Y:][Z:] trail for a given node from sim_start to 
        a particular time

        If no time given, assume the full time range
        """
        if time is None:
            time = self.tmax

        return [self.d[node][dimension][0:time] for dimension in 0,1,2]

def main():
    pass

def interactive_plot(data):
    """
    Generate the MPL data browser for the flock data
    """
    fig = plt.figure()
    ax = axes3.Axes3D(fig)
    lines = [ ax.plot( xs, ys, zs)[0] for xs,ys,zs in data.d ]

    timeax = plt.axes([0.25, 0.1, 0.65, 0.03])
    timeslider = Slider(timeax, 'Time', 0, data.tmax, valinit=0)

    def update(val):
        """
        Update Line display across time
        """
        for n,line in enumerate(lines):
            (xs,ys,zs)=data.trail_of(n,timeslider.val)
            line.set_data(xs,ys)
            line.set_3d_properties(zs)
            line.set_label(data.names[n])
        plt.draw()

    timeslider.on_changed(update)

    shape = data.environment
    ax.set_xlim3d((0,shape[0]))
    ax.set_ylim3d((0,shape[1]))
    ax.set_zlim3d((0,shape[2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
