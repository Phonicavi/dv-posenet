import h5py
import os


class H5(object):
	"""H5 Dataset Base"""
	def __init__(self, base_path, file_name, read_only=True, auto_set=True):
		self.file_path = os.path.join(base_path, file_name)
		self.read_only = read_only
		mode = 'r' if read_only else 'r+'
		try:
			self.group = h5py.File(self.file_path, mode) if auto_set else None
		except Exception as e:
			raise e

	def set(self):
		mode = 'r' if self.read_only else 'r+'
		try:
			self.group = h5py.File(self.file_path, mode)
		except Exception as e:
			raise e
			print('[Exception] H5.set failed')

	def count(self):
		try:
			return len(self.group.get(u"S2d"))
		except Exception as e:
			return None

	def get(self, field):
		try:
			return self.group.get(field)
		except Exception as e:
			return None

	def load(self):
		pass

	def save(self):
		pass


class Input(H5):
	"""Input File: S2d(n,16,2), rank(n,16), rankMat(n,16,16)"""
	def __init__(self, base_path, file_name, read_only=True):
		super(Input, self).__init__(base_path, file_name, read_only, True)
		self.S2d = self.S2d()
		self.rank = self.rank()
		self.rankMat = self.rankMat()
		pass

	def S2d(self):
		return self.get(u"S2d")

	def rank(self):
		return self.get(u"rank")

	def rankMat(self):
		return self.get(u"rankMat")


class Target(H5):
	"""Target File: Idx(n), S2d(n,16,2), S3d(n,16,3), center(n,2), rank(n,16), scale(n)"""
	def __init__(self, base_path, file_name, read_only=True):
		super(Target, self).__init__(base_path, file_name, read_only, True)
		self.Idx = self.Idx()
		self.S2d = self.S2d()
		self.S3d = self.S3d()
		self.center = self.center()
		self.rank = self.rank()
		self.scale = self.scale()
		pass

	def Idx(self):
		return self.get(u"Idx")

	def S2d(self):
		return self.get(u"S2d")

	def S3d(self):
		return self.get(u"S3d")

	def center(self):
		return self.get(u"center")

	def rank(self):
		return self.get(u"rank")

	def scale(self):
		return self.get(u"scale")

