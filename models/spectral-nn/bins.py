from copy import deepcopy
from collections.abc import Iterable
from numbers import Real

class Bins:
	"""
		A Bins class for binning numeric data. 

		Assumptions:
			- edges are lower-bound inclusive, upper-bound exclusive
			- user provides bounding edges (minimum inclusive, 
				maximum exclusive) 
			- if user tries to add values out of bounds,
				the data will be stored in the 'trash' bin
	"""
	inf = float('inf')

	def _hash(self, val):
		"""
			A bad hash function meant for intentionally binning data.
			Assumes the edges are sorted.
			Almost always runs at worst case O(log(n)) where n is
				the number of bin edges.
		"""
		assert(isinstance(val, Real))

		edges = [-self.inf] + self.edges[:] + [self.inf]

		left = 0
		right = len(edges) - 1
		mid = -1
		result = -1

		# iterative binary search
		while left <= right:
			mid = (right + left) // 2

			if edges[left] == val:
				result = left - 1
				break

			elif edges[mid] == val:
				result = mid - 1
				break

			elif edges[right] == val:
				result = right - 1
				break

			elif edges[mid] > val:
				if right == mid:
					result = min(right, left) - 1
					break
				right = mid

			else:
				if left == mid:
					result = min(right, left) - 1
					break
				left = mid

		if result >= len(edges) - 3:
			return -1

		return result

	def __init__(self, edges : list, vals : list = []):
		# safety assertions
		assert(all([isinstance(e, Real) for e in edges]))
		assert(len(edges) > 1)
		assert(all([isinstance(v, Real) for v in vals]))
		
		# store copy of the edges list
		self.edges = edges[:]
		self.edges.sort()

		# number or actual bins
		self.n = len(edges) - 1

		# save bin edges as tuples
		self.bin_tuples = [(edges[i], edges[i+1]) for i in range(self.n)]

		# initialize the bins structure as an indexed dictionary
		self.bins = {i:[] for i in range(self.n)}
		# this is so we can overwrite the -1 index to the 
		# trash bin.
		self.bins[-1] = []

		# explicit attribute for the trash bin
		self.trash = self.bins[-1]

		# add values to the bins
		self.add(vals[:])


	def bounds(self):
		"""
			Returns a list of tuples that represent the edges
			of each bin (len(edges) - 1 elements)
		"""
		return self.bin_tuples

	def add(self, val):
		"""
			Adds a value to its proper bin if a real number,
			if list, adds all items in that list to their proper bins

			Returns the index of the bin the value was added to if 
				adding a single value (-1 for error checking, 
					but value gets added to the trash bin), 
			Returns number of items added if val was a valid iterable
		"""
		# type checking
		assert(isinstance(val, (Real, Iterable)))
		if isinstance(val, Iterable):
			assert(all([isinstance(v, Real) for v in val]))

			# add value to the bins via recursive call
			cnt = 0
			for v in val:
				if self.add(v) < 0:
					continue
				cnt += 1
			return cnt

		index = self._hash(val)
		self.bins[index].append(val)
		return index

	def counts(self, labels : bool = True):
		"""
			Returns the current bin counts
			if labels is True, returns the counts as a 
				dictionary with tuple keys
			otherwise return a list of values whose indices
				correspond to the bin's lower edge
		"""
		counts = [len(self.bins[i]) for i in range(self.n)]
		if labels:
			return dict(zip(self.bin_tuples, counts))
		return counts

	def __repr__(self):
		s = ""
		s += "Bounds: {}\n".format(", ".join([repr(t) for t in self.bin_tuples]))
		s += "Bins:\n"
		for i in range(self.n):
			s += repr(self.bins[i]) + '\n'
		s += "Trash: {}\n".format(self.bins[-1])
		return s

if __name__ == "__main__":
	"""
		Test cases
	"""
	vals = [1, 2, 3, 4, 5, 6, 7, 8]

	bins = Bins([float('-inf'), 4, float('inf')])

	print(bins)

	print(f"adding vals: {vals}")
	bins.add(vals)

	print(bins)

	bins = Bins([2, 4, 7])

	print(bins)

	print(f"adding vals: {vals}")
	bins.add(vals)

	print(bins)

# references
# https://www.geeksforgeeks.org/binary-search/
# https://stackoverflow.com/questions/21361106/
# https://stackoverflow.com/questions/23681948/