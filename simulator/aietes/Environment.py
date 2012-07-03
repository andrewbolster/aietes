import logging
from Tools import baselogger,dotdict
import numpy
import uuid
class Environment():
    """
    Environment Class representing the physical environment inc any objects
    / activities within that environment that are not controlled by the
    simulated entities i.e. wind, tides, speed of sound at depth, etc
    """
    def __init__(self,shape=[100,100,100],resolution=1,base_depth=-1000,sos_model=None):
        """
        Generate a box with points from 0 to (size) in each dimension, where 
        each point represents a cube of side resolution metres:
            Volume is the representation of the physical environment (XYZ)
            Map is the 
        """
        self.logger = logging.getLogger("%s.%s"%(baselogger.name,self.__class__.__name__))
        self.logger.info('creating instance')
        self.volume=numpy.ndarray(shape=shape,dtype=uuid.UUID)
        self.map=dotdict({})
        self.depth=base_depth
        self.sos=1400
        #TODO Random Surface Generation
        #self.generateSurface()
        #TODO 'Tidal motion' factor

    def random_position(self,empty=True):
        """
        Return a random empty map reference within the environment volume
        """
        empty=False
        while not empty:
            ran_x = numpy.random.randint(0,self.volume.shape[0])
            ran_y = numpy.random.randint(0,self.volume.shape[1])
            ran_z = numpy.random.randint(0,self.volume.shape[2])
            empty = not bool(self.volume[ran_x, ran_y, ran_z ])

        return [ran_x, ran_y, ran_z ]

    def update_object(self,object_id,position):
        """
        Update the environment to reflect a movement
        """
        self.logger.debug("Moving %s from %s to %s"%(object_id,
                                                     self.map[object_id],
                                                     position)
                         )
        self.map[object_id]=position


    def export(self,filename=None):
        """
        Export the current environment to a csv
        """
        assert filename is not None
        numpy.savez(filename, self)
        #TODO finish this


