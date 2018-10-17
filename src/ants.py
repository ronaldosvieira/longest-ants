import numpy as np
import pandas as pd
from itertools import chain

class Colony:
	def __init__(self, edges):
		self.V = set(chain(*zip(*edges.index.values)))
		self.E = edges

	def run(self, **params):
		V, E, size = self.V, self.E, len(self.V)
		ph = E.where(E.isnull(), 1 / E.count(axis = 1).sum())

		for i in range(params['max_iter']):
			for k in range(params['ants']):
				pass
