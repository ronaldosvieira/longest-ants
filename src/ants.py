import numpy as np
import pandas as pd
from itertools import chain

class Colony:
	def __init__(self, edges):
		self.V = set(chain(*zip(*edges.index.values)))
		self.E = edges

	def run(self, **params):
		V, E, N = self.V, self.E, max(self.V)
		ph = E.where(E.isnull(), 1 / E.count(axis = 1).sum())
		fitness = (E - E.min()) / (E.max() - E.min())

		best_soln = (- float('inf'), [])

		for i in range(params['max_iter']):
			best_local_soln = (- float('inf'), [])

			probs = ph * params['alpha'] + fitness * params['beta']

			for k in range(params['ants']):
				soln = [1]

				possible = probs.loc[soln[-1]]
				possible = possible[~possible.index.isin(soln + [N])]

				while not possible.empty:
					v = possible.sample(weights = 'weight').index.values[0]

					soln.append(v)

					possible = probs.loc[v]
					possible = possible[~possible.index.isin(soln + [N])]

				while N not in probs.loc[soln[-1]].index:
					soln.pop()

				soln.append(100)

				selected_edges = list(zip(soln, soln[1:]))
				cost = E[E.index.isin(selected_edges)].sum()['weight']

				if cost > best_local_soln[0]:
					best_local_soln = (cost, list(selected_edges))

				if cost > best_soln[0]:
					best_soln = (cost, list(selected_edges))

			in_local_best = ph.index.isin(best_local_soln[1])
			in_global_best = ph.index.isin(best_soln[1])

			ph[~(in_local_best | in_global_best)] *= 1 - params['evap']
			ph[in_local_best | in_global_best] *= 1 + params['evap']

			print(best_soln[0], best_local_soln[0])
			print(ph.mean()['weight'], ph.max()['weight'], ph.min()['weight'])
