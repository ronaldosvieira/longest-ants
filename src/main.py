#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import argparse
import ants

def read_dataset(file):
	return pd.read_csv(file, sep = '\t',
			names = ['u', 'v', 'weight'], index_col = ['u', 'v'])

def get_args():
	parser = argparse.ArgumentParser(
		description = "Ant colony optimization (ACO) \
		for the longest path problem.",
		add_help = True)

	parser.add_argument('dataset', type = open, 
		help = 'which dataset to use')
	parser.add_argument('--ants', '-n', dest = 'ants', type = int,
		default = 10, help = "number of ants to use")
	parser.add_argument('--max-iter', '-i', dest = 'max_iter', type = int,
		default = 50, help = "number of iterations of the algorithm")
	parser.add_argument('--evap', '-ev', dest = 'evap', type = float, 
		default = 0.05, help = "evaporation rate")
	parser.add_argument('-q', dest = 'Q', type = float, 
		default = 0.001, help = "Q parameter")
	parser.add_argument('--alpha', '-a', dest = 'alpha', type = float, 
		default = 1.0, help = "alpha parameter")
	parser.add_argument('--beta', '-b', dest = 'beta', type = float, 
		default = 1.0, help = "beta parameter")
	parser.add_argument('--seed', '-s', dest = 'seed', type = int, 
		default = None, help = "random seed to use")

	try:
		return parser.parse_args()
	except IOError as error:
		parser.error(str(error))

def main():
	args = get_args()

	edges = read_dataset(args.dataset)

	colony = ants.Colony(edges)

	solutions, info = colony.run(**vars(args))

	print(solutions[-1])

	best = list(map(lambda i: i['best'], info))
	mean = zip(map(lambda i: i['mean'], info), map(lambda i: i['std'], info))
	meanstd = list(map(lambda i: '{} +- {:.2f}'.format(*i), mean))
	worst = list(map(lambda i: i['worst'], info))

	print(pd.DataFrame(list(zip(best, worst, meanstd)), 
		columns = ['best', 'worst', 'mean']))

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("Stopping")
