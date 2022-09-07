import os
import time
import config
import os.path
from train_classifier import train
from generate_features import generate_features
from merge import run_merge
from refine import run_refine
from clustering import run_kmeans
from ood_detection import generate_ood_masks

class StageClassifier():

	def __init__(self, step, conf, root_path, save_conf):
		self.step = step
		self.conf = conf
		self.root_path = root_path
		self.save_conf = save_conf

	def execute(self):
		st = time.time()
		train(self.step, self.conf, self.save_conf)
		et = time.time()
		print('Total execution time: {0:.2f} s'.format(et-st))
		print(f'Step {self.step}, stage classifier completed at {os.path.join(self.root_path,f"step{self.step}","classifier")}')


class StageGenerateFeatures():

	def __init__(self, step, conf, root_path, save_conf):
		self.step = step
		self.conf = conf
		self.root_path = root_path
		self.save_conf = save_conf

	def execute(self):
		st = time.time()
		generate_features(self.step, self.conf, self.save_conf)
		et = time.time()
		print('Total execution time: {0:.2f} s'.format(et-st))
		print(f'Step {self.step}, stage generate features completed at {os.path.join(self.root_path,f"step{self.step}","generate_features")}')


class StageOOD():

	def __init__(self, step, conf, root_path, save_conf):
		self.step = step
		self.conf = conf
		self.root_path = root_path
		self.save_conf = save_conf

	def execute(self):
		st = time.time()
		generate_ood_masks(self.step, self.conf, self.save_conf)
		et = time.time()
		print('Total execution time: {0:.2f} s'.format(et-st))
		print(f'Step {self.step}, stage OOD detection completed at {os.path.join(self.root_path,f"step{self.step}","ood_detection")}')


class StageClustering():

	def __init__(self, step, conf, root_path, save_conf):
		self.step = step
		self.conf = conf
		self.root_path = root_path
		self.save_conf = save_conf

	def execute(self):
		st = time.time()
		run_kmeans(self.step, self.conf, self.save_conf)
		et = time.time()
		print('Total execution time: {0:.2f} s'.format(et-st))
		print(f'Step {self.step}, stage clustering completed at {os.path.join(self.root_path,f"step{self.step}","clustering")}')


class StageMerge():

	def __init__(self, step, conf, root_path, save_conf):
		self.step = step
		self.conf = conf
		self.root_path = root_path
		self.save_conf = save_conf

	def execute(self):
		st = time.time()
		run_merge(self.step, self.conf, self.save_conf)
		et = time.time()
		print('Total execution time: {0:.2f} s'.format(et-st))
		print(f'Step {self.step}, stage merging completed at {os.path.join(self.root_path,f"step{self.step}","merge")}')


class StageRefine():

	def __init__(self, step, conf, root_path, save_conf):
		self.step = step
		self.conf = conf
		self.root_path = root_path
		self.save_conf = save_conf

	def execute(self):
		st = time.time()
		run_refine(self.step, self.conf, self.save_conf)
		et = time.time()
		print('Total execution time: {0:.2f} s'.format(et-st))
		print(f'Step {self.step}, stage refining completed at {os.path.join(self.root_path,f"step{self.step}","refine")}')
