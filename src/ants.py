import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict

class Colony:
	def __init__(self, edges):
		self.V = set(chain(*zip(*edges.index.values)))
		self.E = edges

	def run(self, **params):
		V, E, N = self.V, self.E, max(self.V)
		ph = E.where(E.isnull(), 1)
		fitness = E

		solutions = []
		info = []

		best_soln = (- float('inf'), [])

		for i in range(params['max_iter']):
			best_local_soln = (- float('inf'), [])

			probs = ph ** params['alpha'] + fitness ** params['beta']

			ants = []
			repeated_edges = defaultdict(lambda: -1)

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

				soln.append(N)

				selected_edges = list(zip(soln, soln[1:]))
				cost = E[E.index.isin(selected_edges)].sum()['weight']

				for edge in selected_edges:
					repeated_edges[edge] += 1

				ants.append((selected_edges, cost))

				if cost > best_local_soln[0]:
					best_local_soln = (cost, list(selected_edges))

				if cost > best_soln[0]:
					best_soln = (cost, list(selected_edges))

			solutions.append(pd.DataFrame(ants, 
				columns = ['soln', 'cost']))

			info.append({
				'best': solutions[-1]['cost'].max(),
				'mean': solutions[-1]['cost'].mean(),
				'worst': solutions[-1]['cost'].min(),
				'repeated_edges': sum(list(repeated_edges.values()))}
			)

			in_local_best = ph.index.isin(best_local_soln[1])
			in_global_best = ph.index.isin(best_soln[1])

			ph *= 1 - params['evap']

			ph[in_local_best] += params['Q'] * best_local_soln[0]
			ph[in_global_best] += params['Q'] * best_soln[0]

			print(best_soln[0], best_local_soln[0], solutions[-1]['cost'].mean())
			print(probs.mean()['weight'], probs.max()['weight'], probs.min()['weight'])

		return solutions, info
