import numpy
class Environment(numpy.ndarray):
    """
    Environment Class representing the physical environment inc any objects
    / activities within that environment that are not controlled by the
    simulated entities i.e. wind, tides, speed of sound at depth, etc
    """
    def __init__(self,shape=[100,100,100],scale=1,base_depth=-1000,sos_model=None):
        """
        Generate a box with points from 0 to (size) in each dimension, where 
        each point represents a cube of side scale metres
        """
        numpy.ndarray.__init__(self,shape=shape,dtype=numpy.float)
        self.depth=base_depth
        self.sos=1400
        #TODO Random Surface Generation
        self.generateSurface()
        #TODO 'Tidal motion' factor

    def export(self,filename=None):
        """
        Export the current environment to a csv
        """
        assert filename is not None
        numpy.savez(filename, self)
        #TODO finish this


