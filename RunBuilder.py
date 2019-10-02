from collections import OrderedDict
from collections import namedtuple
from itertools import product

class RunBuilder():
	'''
	Returns a named tuple to use for training
	Input: OrderedDict() having array of values for a given key
	Output: list of named tuples
	'''
	@staticmethod
	def get_runs(params):
		'''
		a static method which takes OrderedDict as input and returns a list of named tuples.
		'''
		Run = namedtuple('Run',params.keys())


		runs = []

		for v in product(*params.values()):
			runs.append(Run(*v))

		return runs


if __name__ == 'main':
	
	params = OrderedDict(
		lr = [0.01, 0.001],
		batch_size = [1000, 10000],
		device = ['cuda', 'cpu'],
		shuffle = [True, False]
	)

	for run in RunBuilder.get_runs(params):
		print(run)
		
