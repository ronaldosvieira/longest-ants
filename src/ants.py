import numpy as np
import pandas as pd
from itertools import chain

class Colony:
	def __init__(self, edges):
		self.V = set(chain(*zip(*edges.index.values)))
		self.E = edges

	def run(self, **params):
		V, E = self.V, self.E
		ph = E.where(E.isnull(), 1 / E.count(axis = 1).sum())

		best_soln = (- float('inf'), [])

		for i in range(params['max_iter']):
			for k in range(params['ants']):
				soln = list(np.random.choice(E.index.values))

				possible = ph.loc[soln[-1]]
				possible = possible[~possible.index.isin(soln)]

				while not possible.empty:
					v = possible.sample(weights = 'weight').index.values[0]

					soln.append(v)

					possible = ph.loc[v]
					possible = possible[~possible.index.isin(soln)]

				selected_edges = list(zip(soln, soln[1:]))
				cost = E[E.index.isin(selected_edges)].sum()['weight']

				if cost > best_soln[0]:
					best_soln = (cost, list(selected_edges))

		print(best_soln)
