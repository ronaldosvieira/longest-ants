#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def read_dataset(name):
	edges = pd.read_csv('data/{}'.format(name), sep = '\t',
			names = ['u', 'v', 'weight'])

	W = pd.crosstab(edges.u, edges.v, values = edges.weight, 
			aggfunc = lambda x: x).replace({np.nan: None})

	return W
