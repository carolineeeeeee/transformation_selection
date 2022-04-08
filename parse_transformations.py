# a collection of transformations
from utils import *
import pandas as pd
import json
from pattern_matching import *
#import itertools
#import math 
#from nltk.corpus import stopwords
#import yake


class Transformation:
	"""A class for transformation information"""

	# TODO: later update format
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

	def match_keywords(self):
		# match keywords
		all_keywords = []
		transformations_keywords = {}
		#self.all_transformations = [t for t in self.all_transformations if t.index in [1, 2, 3]]
		for transf in self.all_transformations:
			#if 'Crop And Pad' not in transf.name:
			#	continue
			#if transf.name != 'Median Blur':
			#	continue
			print(transf.name)
			print(transf.description)
			#print(type(transf.description))

			transformations_keywords[transf.index] = []

			#parsed_results = list(set(itertools.chain.from_iterable(parse_transf(transf))))
			parsed_results = parse_transf(transf)
			transf.match = parsed_results
			print(parsed_results)
			#continue
		'''
		
			# first find keywords for names
			t_keywords = find_keywords(parsed_results, keywords, is_transf=True)
			print(t_keywords)

			transformations_keywords[transf.index] += t_keywords
			all_keywords += t_keywords
			# then find keywords for descriptions using parsed text

		# remove keywords not meaningful (appear in at least half of the transformations)
		to_remove = []
		for k in set(all_keywords):
			if len([t for t in transformations_keywords if k in transformations_keywords[t]]) >= 0.4 * len(self.all_transformations):
				to_remove.append(k)
		for transf in self.all_transformations:
			if transf.index in transformations_keywords:
				new_value = [w for w in transformations_keywords[transf.index] if w not in to_remove]
				transf.keywords = set(new_value)
				transformations_keywords[transf.index] = set(new_value)
		#print(transformations_keywords['Random Rain'])
		#print(transformations_keywords['Advanced Blur'])
		#print(transformations_keywords['Random Snow'])
		#return transformations
		'''
	
	def extract_terms(self):
		# we should ignore descriptions of the values and examples 
		# description of the values should be used to determine the guideword
		# also remove dfault is
		kw_extractor = yake.KeywordExtractor(top=10, stopwords=stopwords.words('english'))
		for t in self.all_transformations:
			print('------------' + t.name + '-------------')
			print('Keywords (old):')
			print(t.keywords)
			print('Our match:')
			print(t.match)
			full_text = t.name + ': ' + t.description + ', \n'
			for param in t.parameters:
				full_text += param + ': ' + t.parameters[param] + '\n'
			print(full_text)
			keywords = kw_extractor.extract_keywords(full_text)
			for kw, v in keywords:
				print("Keyphrase: ",kw, ": score", v)
			print('--------------------------------------------')
		#return 0