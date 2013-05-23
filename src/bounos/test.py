"""Unit tests for bounos"""

import unittest
import bounos
import tempfile
import numpy as np

class DataPackageCreation(unittest.TestCase):
    #TODO TMax testing
    def testSourceParsing(self):
        """ Ensure that DataPackage is correctly created by using a sourcefile: 
            does NOT test numerical validitity of anything
        """
        #Use tempfile to make up an arbitrary name, which is then automatically
        #  suffixed with .npz
        source_file_obj = tempfile.NamedTemporaryFile(delete = False)
        source_file = source_file_obj.name
        source_file_obj.close()
        print source_file
        s_n_nodes = 10
        s_tmax = 1000
        positions = []
        vectors = []
        names = []
        for n in range(s_n_nodes):
            positions.append(np.empty((3,s_tmax)))
            vectors.append(np.empty((3,s_tmax)))
            names.append("testname")

        contributions= [ [ None for _ in range(s_tmax) ] for _ in range(s_n_nodes)],
        achievements = [ [] for _ in range(s_n_nodes)]
        environment = np.empty(3, dtype = int),
        config = {'testconfig': True},
        title = "TESTCONFIG"

        np.savez(source_file,
                positions=positions,
                vectors=vectors,
                names=names,
                contributions=contributions,
                achievements=achievements,
                environment=environment,
                config=config,
                title=title
                )
        dp = bounos.DataPackage(source=source_file+'.npz')
        print dp.p.shape
        self.assertEqual(dp.title, title)
        self.assertEqual(dp.n , s_n_nodes)
        self.assertEqual(dp.tmax , s_tmax)