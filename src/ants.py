import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict

class Colony:
	def __init__(self, edges):
		self.V = set(chain(*zip(*edges.index.values)))
		self.E = edges
		self.N = max(self.V)

	def build_path(self, probs):
		# initializes solution as a list of vertexes
		path = [1]

		# calculates chooseable edges
		# i.e. (u, v) such that u = last visited vertex 
		#      AND v is not already on path
		#      AND v is not vertex N
		possible = probs.loc[1]
		possible = possible[~possible.index.isin(path + [self.N])]

		# while there are edges to take
		while not possible.empty:
			# choose from possible edges probabilistically
			# (normalization is made here)
			v = possible.sample(weights = 'weight').index.values[0]

			# adds edge to solution
			path.append(v)

			# recalculates chooseable edges
			possible = probs.loc[v]
			possible = possible[~possible.index.isin(path + [self.N])]

		# at this time we should have a path from 1 to an
		# arbitrary vertex

		# pops vertexes from the solution until we are
		# able to plug in an edge to N at the end.
		# worst case: path will be empty
		while self.N not in probs.loc[path[-1]].index:
			path.pop()

		# adds edge to N
		path.append(self.N)

		# transforms from list of vertexes to an actual 
		# list of edges
		path = list(zip(path, path[1:]))

		return path

	def run(self, **params):
		solutions = []
		stats = []

		try:
			# sets random seed
			if 'seed' in params:
				np.random.seed(params['seed'])

			# V = list of #s
			# E = (u, v) -> w
			V, E = self.V, self.E

			# initializes all pheromones as 1
			ph = E.where(E.isnull(), 1)

			# sets fitness values to edge weight
			fitness = E

			# initializes best global solution
			gbest = (- float('inf'), [])

			# for each iteration
			for i in range(params['max_iter']):
				# initializes local best solution
				lbest = (- float('inf'), [])

				# initializes data structure for repeated edge checking
				repeated_edges = defaultdict(lambda: -1)

				# tau^alpha * rho^beta
				probs = ph ** params['alpha'] + fitness ** params['beta']

				# initialize list of solutions
				ants = []

				# for each ant
				for k in range(params['ants']):
					# builds a path probabilistically
					path = self.build_path(probs)

					# calculates total path cost
					cost = E[E.index.isin(path)].sum()['weight']

					# saves created solution
					ants.append((path, cost))

					# if better than local best, updates it
					if cost > lbest[0]:
						lbest = (cost, list(path))

					# if better than global best, updates it
					if cost > gbest[0]:
						gbest = (cost, list(path))

					# counts use of edges
					for edge in path:
						repeated_edges[edge] += 1

				# transforms into dataframe
				ants = pd.DataFrame(ants, columns = ['path', 'cost'])

				# gather statistics
				stats.append({
					'best': ants['cost'].max(),
					'worst': ants['cost'].min(),
					'mean': ants['cost'].mean(),
					'std': ants['cost'].std(),
					'size': ants['path'].apply(len).mean() + 1,
					'rep': sum(list(repeated_edges.values())),
					'lbest': lbest[1]
				})

				# checks which edges are in local and global best
				in_local_best = ph.index.isin(lbest[1])
				in_global_best = ph.index.isin(gbest[1])

				# evaporates pheromones
				ph *= 1 - params['evap']

				# adds pheromones to edges in local and global best
				ph[in_local_best] += params['Q'] * lbest[0]
				ph[in_global_best] += params['Q'] * gbest[0]
		except KeyboardInterrupt:
			pass

		return pd.DataFrame(stats)
