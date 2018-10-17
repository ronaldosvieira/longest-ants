#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import ants

def read_dataset(name):
	edges = pd.read_csv('data/{}.txt'.format(name), sep = '\t',
			names = ['u', 'v', 'weight'], index_col = ['u', 'v'])

	'''W = pd.crosstab(edges.u, edges.v, values = edges.weight, 
			aggfunc = lambda x: x).replace({np.nan: None})'''

	return edges

edges = read_dataset('graph1')

colony = ants.Colony(edges)

colony.run(ants = 10, max_iter = 1)