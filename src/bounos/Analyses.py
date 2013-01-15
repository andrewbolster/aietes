__author__ = 'andrewbolster'

from __init__ import DataPackage


def Find_Convergence(data, *args, **kwargs):
	"""
	Return the Time of estimated convergence of the fleet along with some certainty value
	using the Average of Inter Node Distances metric
	i.e. the stability of convergence
	"""
	assert isinstance(data, DataPackage)



