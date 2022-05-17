# a collection of transformations
from utils import *
import pandas as pd
import json
from pattern_matching import *

class Transformation:
	"""A class for transformation information"""
	def __init__(self, index, row):
		self.keywords = []
		self.index = index
		self.row = row
		self.name = row['Name']
		self.source = row['Source']
		if isinstance(row['Description'], str):
			self.description = row['Description']
		else:
			self.description = ''
		self.parameters = json.loads(row['Patameters'])
		self.match = []
	
	def __str__(self):
		results = ''
		results += 'name: ' + self.name + ', \n'
		results += 'source: ' + self.source + ', \n'
		results += 'description: ' + self.description + ', \n'
		results += 'parameters: ' + str(self.parameters)
		return results

class TransformationList:
	"""A class for a library of transformations"""
	def __init__(self, filename, lib_name=None):
		self.filename = filename
		self.all_transformations = []
		self.lib_name = lib_name
		self.parse_transformations(self.filename)

	def parse_transformations(self, filename):
		transformaton_df = pd.read_csv(filename)
		for index, row in transformaton_df.iterrows():
			if self.lib_name:
				if row['Source'] == self.lib_name:
					trans_entry = Transformation(index, row)
					self.all_transformations.append(trans_entry)
			else:
				trans_entry = Transformation(index, row)
				self.all_transformations.append(trans_entry)

	def parse_effect_action(self):
		# match keywords
		transformations_keywords = {}
		for transf in self.all_transformations:
			print(transf.name)
			print(transf.description)

			transformations_keywords[transf.index] = []

			parsed_results = parse_transf(transf)
			transf.match = parsed_results
			print(parsed_results)
			